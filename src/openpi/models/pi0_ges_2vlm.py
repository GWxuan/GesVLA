import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model_2vlm as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.models.tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")
import numpy as np

PALIGEMMA_EOS_TOKEN = 1

def seg_positions(seg_mask, offset):
    """Compute segment-local positions with a fixed offset for masked sequences."""
    # seg_mask: [B, L] bool
    pos = jnp.cumsum(seg_mask, axis=1) - 1  # padding yields -1
    pos = jnp.where(seg_mask, pos + offset, 0)  # padding positions set to 0 (masked out anyway)
    return pos

def _as_bt12(x, b):
    """Convert gesture data to [b, t, 12]; return zeros if input is None."""
    if x is None:
        logger.warning("hand_pose is None, returning zero tensor with shape [b, 16, 12]")
        return jnp.zeros((b, 16, 12), dtype=jnp.float32)

    if hasattr(x, 'shape') and len(x.shape) >= 2:
        return x

    # Otherwise, return a zero tensor.
    return jnp.zeros((b, 8, 12), dtype=jnp.float32)
    
def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def put_along_last_axis(arr, indices, values):
    """Like np.put_along_axis(..., axis=-1), since jax is missing it."""
    assert arr.ndim == indices.ndim == values.ndim, (arr.ndim, indices.ndim, values.ndim)
    onehot = jax.nn.one_hot(indices, arr.shape[-1], dtype=values.dtype)
    put_mask = jnp.einsum("...i,...in->...n", jnp.ones(values.shape, jnp.int32), onehot)
    put_values = jnp.einsum("...i,...in->...n", values, onehot)
    return jnp.where(put_mask, put_values, arr)


def _concat_list(xs, axis=1):
    xs = [x for x in xs if x is not None]
    return jnp.concatenate(xs, axis=axis) if len(xs) else None

def _build_base_attn_mask(input_mask, ar_mask):
    """Build the base attention mask using the standard make_attn_mask logic."""
    return make_attn_mask(input_mask, ar_mask)

def _apply_stream_constraints(
    attn_mask,  # [B, T, S]
    input_mask, # [B, T]
    ranges,      # dict of name -> (start, end)
    *,
    allow_vlm1_to_vlm0: bool = True,
):
    """Apply stream-specific visibility constraints to the attention mask."""
    # Valid mask (padding).
    valid = input_mask[:, None, :] * input_mask[:, :, None]
    attn_mask = jnp.logical_and(attn_mask, valid)

    v1s, v1e = ranges["vlm1"]
    as_, ae = ranges["act"]
    v0s, v0e = ranges["vlm0"]

    # 1) VLM0 does not attend to VLM1 / action.
    attn_mask = attn_mask.at[:, v0s:v0e, v1s:ae].set(False)

    # 2) Action expert does not attend to VLM0.
    attn_mask = attn_mask.at[:, as_:ae, v0s:v0e].set(False)

    # 3) VLM1 does not attend to action (keep the original prefix semantics).
    attn_mask = attn_mask.at[:, v1s:v1e, as_:ae].set(False)

    # 4) VLM1 can attend to VLM0 (asymmetric interaction).
    if allow_vlm1_to_vlm0:
        # Force-enable VLM1->VLM0 attention (still constrained by padding).
        v = valid[:, v1s:v1e, v0s:v0e]
        attn_mask = attn_mask.at[:, v1s:v1e, v0s:v0e].set(v)

    return attn_mask

@dataclasses.dataclass(frozen=True)
class Pi0GesConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    gesture_vlm_variant: _gemma.Variant = "gemma_2b"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 150


    # diffusion loss coefficient
    # L = L_text + diffusion_loss_coeff * L_diffusion
    diffusion_loss_coeff: float = 1.0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0_GES

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Ges":
        return Pi0Ges(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.FuseObservation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.FuseObservation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                ges_images={
                    "ges_0_rgb": image_spec,
                    "ges_1_rgb": image_spec,
                    "ges_2_rgb": image_spec,
                },
                ges_image_masks={
                    "ges_0_rgb": image_mask_spec,
                    "ges_1_rgb": image_mask_spec,
                    "ges_2_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                token_ar_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                token_loss_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
                diffusion_loss_mask=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0Ges(_model.BaseModel):
    def __init__(self, config: Pi0GesConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # gesture_variant = config.gesture_vlm_variant or config.paligemma_variant
        gesture_vlm_config = _gemma.get_config(config.gesture_vlm_variant)

        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                # expert0: VLM1 (load ckpt)
                # expert1: action expert (existing)
                # expert2: VLM0 (gesture VLM)
                configs=[paligemma_config, action_expert_config, gesture_vlm_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # NEW: Two-layer MLP for gesture sequences (b,t,12) → (b,t,width).
        self.hand_pose_mlp_1  = nnx.Linear(12, 256, rngs=rngs)
        self.hand_pose_mlp_2 = nnx.Linear(256, paligemma_config.width, rngs=rngs) 

        self.diffusion_loss_coeff = config.diffusion_loss_coeff


    @at.typecheck
    def embed_img_txt_dual(
        self, obs: _model.FuseObservation
    ) -> tuple[
        at.Float[at.Array, "b s1 emb"], at.Bool[at.Array, "b s1"], at.Int[at.Array, "b s1"],  # VLM1
        at.Float[at.Array, "b s0 emb"], at.Bool[at.Array, "b s0"], at.Int[at.Array, "b s0"],  # VLM0
    ]:
        """
        VLM1: images only (obs.images), NO text.
        VLM0: ges_images (obs.ges_images) + hand (between) + text.
        """
        b = obs.tokenized_prompt.shape[0]

        # ---------------------------
        # VLM1 image embeddings: obs.images
        # ---------------------------
        v1_img_tokens = []
        v1_img_mask = []
        v1_img_ar = []

        for name in obs.images:
            image_emb, _ = self.PaliGemma.img(obs.images[name], train=False)
            v1_img_tokens.append(image_emb)
            v1_img_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_emb.shape[1]))
            v1_img_ar.append(jnp.zeros_like(v1_img_mask[-1], dtype=jnp.int32))

        v1_img_tokens = jnp.concatenate(v1_img_tokens, axis=1)
        v1_img_mask = jnp.concatenate(v1_img_mask, axis=1)
        v1_img_ar = jnp.concatenate(v1_img_ar, axis=1)

        vlm1_tokens = v1_img_tokens
        vlm1_mask = v1_img_mask
        vlm1_ar = v1_img_ar

        # ---------------------------
        # VLM0 image embeddings: obs.ges_images
        # ---------------------------
        ges_images = getattr(obs, "ges_images", None)
        # print(obs.keys())
        if ges_images is None:
            raise AttributeError("obs.ges_images is required for VLM0.")

        ges_image_masks = getattr(obs, "ges_image_masks", None)

        v0_img_tokens = []
        v0_img_mask = []
        v0_img_ar = []

        for name in ges_images:
            image_emb, _ = self.PaliGemma.img(ges_images[name], train=False)
            v0_img_tokens.append(image_emb)

            if ges_image_masks is not None and name in ges_image_masks and ges_image_masks[name] is not None:
                m = ges_image_masks[name]
            else:
                m = jnp.ones((b,), dtype=jnp.bool_)

            v0_img_mask.append(einops.repeat(m, "b -> b s", s=image_emb.shape[1]))
            v0_img_ar.append(jnp.zeros_like(v0_img_mask[-1], dtype=jnp.int32))

        v0_img_tokens = jnp.concatenate(v0_img_tokens, axis=1)
        v0_img_mask = jnp.concatenate(v0_img_mask, axis=1)
        v0_img_ar = jnp.concatenate(v0_img_ar, axis=1)

        # ---------------------------
        # VLM0 hand embeddings: INSERT BETWEEN image and text
        # ---------------------------
        hand_pose_seq = getattr(obs, "hand_pose", None)
        hand_pose_mask = getattr(obs, "hand_pose_mask", None)

        if hand_pose_seq is not None and hand_pose_mask is not None:
            hand_pose_seq = _as_bt12(hand_pose_seq, b)  # [b,t,12]

            # Mask fallback: if the field exists but is None, use all ones.
            # if hasattr(obs, "hand_pose_mask") and (obs.hand_pose_mask is not None):
            hp_mask = obs.hand_pose_mask.astype(jnp.bool_)  # [b,t]
            # else:
            #     hp_mask = jnp.ones(hand_pose_seq.shape[:2], dtype=jnp.bool_)

            hp_emb = self.hand_pose_mlp_1(hand_pose_seq)  # [b,t,256]
            hp_emb = nnx.swish(hp_emb)
            hp_emb = self.hand_pose_mlp_2(hp_emb)         # [b,t,emb]
            hp_ar = jnp.zeros_like(hp_mask, dtype=jnp.int32)
        else:
            hp_emb = None
            hp_mask = None
            hp_ar = None

        # ---------------------------
        # VLM0 text embeddings (still needed)
        # ---------------------------
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required for VLM0"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required for VLM0"
        assert obs.token_ar_mask is not None, "Token ar mask is required for VLM0"

        txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        txt_mask = obs.tokenized_prompt_mask
        txt_ar = obs.token_ar_mask.astype(jnp.int32)

        
        # ---------------------------
        # Assemble VLM0 = ges_img + hand + text   (hand inserted between)
        # ---------------------------
        if hp_emb is not None:
            vlm0_tokens = jnp.concatenate([v0_img_tokens, hp_emb, txt_emb], axis=1)
            vlm0_mask = jnp.concatenate([v0_img_mask, hp_mask, txt_mask], axis=1)
            vlm0_ar = jnp.concatenate([v0_img_ar, hp_ar, txt_ar], axis=1)
        else:
            vlm0_tokens = jnp.concatenate([v0_img_tokens, txt_emb], axis=1)
            vlm0_mask = jnp.concatenate([v0_img_mask, txt_mask], axis=1)
            vlm0_ar = jnp.concatenate([v0_img_ar, txt_ar], axis=1)

        return vlm1_tokens, vlm1_mask, vlm1_ar, vlm0_tokens, vlm0_mask, vlm0_ar

    @at.typecheck
    def embed_state_action(
        self, obs: _model.FuseObservation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask, dtype=jnp.int32)
        ar_mask = jnp.broadcast_to(ar_mask,
                                   (tokens.shape[0], tokens.shape[1]))
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.FuseObservation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train, image_keys=list(observation.images.keys()),tokenizer=_tokenizer.FusePaligemmaTokenizer)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of img + txt prefix + txt suffix + state + action
        # img_txt_tokens, img_txt_mask, img_txt_ar_mask = self.embed_img_txt(observation)
        # state_action_tokens, state_action_mask, state_action_ar_mask = self.embed_state_action(observation, x_t, time)
        # input_mask = jnp.concatenate([img_txt_mask, state_action_mask], axis=1)
        # ar_mask = jnp.concatenate([img_txt_ar_mask, state_action_ar_mask], axis=1)
        # attn_mask = make_attn_mask(input_mask, ar_mask)
        # positions = jnp.cumsum(input_mask, axis=1) - 1
        # (img_txt_pre_logits, state_action_out), _ = self.PaliGemma.llm(
        #     [img_txt_tokens, state_action_tokens], mask=attn_mask, positions=positions
        # )

        vlm1_tokens, vlm1_mask, vlm1_ar, vlm0_tokens, vlm0_mask, vlm0_ar = self.embed_img_txt_dual(observation)
        act_tokens, act_mask, act_ar = self.embed_state_action(observation, x_t, time)

        # total concatenation order must match embedded list order: [VLM1, ACT, VLM0]
        input_mask = jnp.concatenate([vlm1_mask, act_mask, vlm0_mask], axis=1)
        ar_mask = jnp.concatenate([vlm1_ar, act_ar, vlm0_ar], axis=1)

        base_attn = make_attn_mask(input_mask, ar_mask)

        v1_len = vlm1_mask.shape[1]
        a_len = act_mask.shape[1]
        v0_len = vlm0_mask.shape[1]

        ranges = {
            "vlm1": (0, v1_len),
            "act":  (v1_len, v1_len + a_len),
            "vlm0": (v1_len + a_len, v1_len + a_len + v0_len),
        }
        attn_mask = _apply_stream_constraints(base_attn, input_mask, ranges, allow_vlm1_to_vlm0=True)

        # positions = jnp.cumsum(input_mask, axis=1) - 1


        # Goal: VLM0 offset=0, VLM1 offset=v0_valid_len? (fixed length is simpler: use v0_len).
        # Fixed offsets keep shapes static and JIT-friendly.
        pos_v0 = seg_positions(vlm0_mask, 0)
        pos_v1 = seg_positions(vlm1_mask, v0_len)
        pos_a  = seg_positions(act_mask,  v0_len + v1_len)

        # Note: final positions must match the real concatenation order [VLM1, ACT, VLM0].
        positions = jnp.concatenate([pos_v1, pos_a, pos_v0], axis=1)

        (vlm1_pre_logits, act_out, vlm0_pre_logits), _ = self.PaliGemma.llm(
            [vlm1_tokens, act_tokens, vlm0_tokens],
            mask=attn_mask,
            positions=positions,
        )

        txt_loss = 0

        # action diffusion loss
        v_t = self.action_out_proj(act_out[:, -self.action_horizon :])
        action_loss = jnp.mean(
            (
            jnp.square(v_t - u_t) *
            observation.diffusion_loss_mask[:, None, None]
            ),
            axis=(-2,-1))

        info = {
            'text_loss': txt_loss,
            'action_loss': action_loss,
        }
        

        loss = txt_loss + self.diffusion_loss_coeff * action_loss
        return loss, info
    
    @at.typecheck
    def prefill(self,
                rng: at.KeyArrayLike,
                observation: _model.FuseObservation,
                *,
                temprature: float = 0.,
                ) -> tuple[
                    _model.FuseObservation,
                    _gemma.KVCache,
                    at.Int[at.Array, "b 1"],
                    at.Float[at.Array, "b 1 v"],
                    at.Bool[at.Array, "b s"],
                    at.Int[at.Array, "b s"],
                    at.Bool[at.Array, "b"],
                    at.Int[at.Array, ""],  # v1_len: number of VLM1 tokens for act() masking.
                ]:
        """
        Prefill the prefix. Used for policy serving.

        Args:
            rng: PRNG key.
            observation: FuseObservation.
                `tokenized_suffix` and `tokenized_suffix_mask` should be None.
            temprature: decoding temprature.
        
        Returns:
            tuple: A tuple containing
                - kv_cache: KV cache for the prefix. `<END_OF_PREFIX>`token included.
                - token: the next token of `<END_OF_PREFIX>`.
                - eop_logit: logit for the `<END_OF_PREFIX>` token.
                - prefix_mask: input mask for the prefix.
                - prefix_positions: position id for the prefix.
                - to_act: whether to act or to reason.
        """
        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))
        # assert jnp.any(observation.token_ar_mask == 1, axis=-1).all(), "mask must include suffix"
        # locate suffix start position
        first_one_indices = jnp.argmax(observation.token_ar_mask, axis=-1)
        padding_mask = jnp.arange(observation.token_ar_mask.shape[-1]) >= first_one_indices[..., jnp.newaxis]
        # padding suffix to 0
        observation = dataclasses.replace(observation, 
                                          tokenized_prompt=jnp.where(padding_mask, 0, observation.tokenized_prompt),
                                          tokenized_prompt_mask=jnp.logical_not(padding_mask),)

        # embed inputs
        # prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_img_txt(observation)
        # prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # # first fill KV cache with a forward pass of the prefix (img + txt prefix)
        # (pre_logit, _), kv_cache = self.PaliGemma.llm(
        #     [prefix_token_embeddings, None], mask=prefix_attn_mask, positions=prefix_positions
        # )
        
        # according to prefix mask, the last token is <END_OF_PREFIX>, find its logit
        # eop_indices = prefix_positions[:,-1]
        # eop_pre_logit = jnp.take_along_axis(pre_logit, eop_indices[:, None, None], axis=1)
        # eop_logit = self.PaliGemma.llm(eop_pre_logit, method="embedder_decode")

        vlm1_tokens, vlm1_mask, vlm1_ar, vlm0_tokens, vlm0_mask, vlm0_ar = self.embed_img_txt_dual(observation)

        # prefix only, no action expert tokens
        input_mask = jnp.concatenate([vlm1_mask, vlm0_mask], axis=1)
        ar_mask = jnp.concatenate([vlm1_ar, vlm0_ar], axis=1)

        base_attn = make_attn_mask(input_mask, ar_mask)

        v1_len = vlm1_mask.shape[1]
        v0_len = vlm0_mask.shape[1]
        ranges = {"vlm1": (0, v1_len), "act": (v1_len, v1_len), "vlm0": (v1_len, v1_len + v0_len)}
        attn_mask = _apply_stream_constraints(base_attn, input_mask, ranges, allow_vlm1_to_vlm0=True)

        # RoPE positions follow training-time offsets.
        # Training: VLM0 offset=0, VLM1 offset=v0_len, Action offset=v0_len+v1_len.
        # Prefill has no Action, but VLM0/VLM1 offsets must still match training.
        # Plain cumsum makes VLM1 start at 0 and VLM0 follow, opposite to training.
        pos_v1 = seg_positions(vlm1_mask, v0_len)   # VLM1 starts at v0_len.
        pos_v0 = seg_positions(vlm0_mask, 0)          # VLM0 starts at 0.
        # Concatenation order matches the physical token order [VLM1, VLM0].
        positions = jnp.concatenate([pos_v1, pos_v0], axis=1)

        (pre_logit, _none, _vlm0_out), kv_cache = self.PaliGemma.llm(
            [vlm1_tokens, None, vlm0_tokens],
            mask=attn_mask,
            positions=positions,
        )

        # eop_indices still refer to the last token in the full prefix.
        # EOP belongs to the end of VLM0 text, so use the last valid VLM0 token.
        # pre_logit is VLM1 output (only v1_len tokens); indexing v1+v0-1 would clip
        # to the last VLM1 image token, which was never trained to predict text.
        vlm0_last_valid_idx = jnp.sum(vlm0_mask, axis=1) - 1  # [b]
        eop_pre_logit = jnp.take_along_axis(
            _vlm0_out, vlm0_last_valid_idx[:, None, None], axis=1)  # [b, 1, emb]
        eop_logit = self.PaliGemma.llm(eop_pre_logit, method="embedder_decode")

        prefix_mask = input_mask
        prefix_positions = positions


        # decide to think or to act
        # only sample between _tokenizer.BEGIN_OF_ACTION
        # and _tokenizer.BEGIN_OF_REASONING
        valid_tokens = jnp.array([_tokenizer.BEGIN_OF_ACTION,
                                  _tokenizer.BEGIN_OF_REASONING])
        valid_mask = jnp.full((1, 1, eop_logit.shape[-1]), -jnp.inf)
        valid_mask = valid_mask.at[:, :, valid_tokens].set(0)
        eop_logit = eop_logit + valid_mask
        if temprature > 0.0:
            token = jax.random.categorical(rng, eop_logit / temprature, axis=-1)
        else:
            token = jnp.argmax(eop_logit, axis=-1)
        has_boa = jnp.any(
            token == _tokenizer.BEGIN_OF_ACTION,
            axis=1)
        
        # v1_len is used to build the action-specific prefix mask in act().
        return observation, kv_cache, token, eop_logit, prefix_mask, prefix_positions, has_boa, jnp.array(v1_len, dtype=jnp.int32)
    
    @at.typecheck
    def reason(self,
              rng: at.KeyArrayLike,
              last_logit: at.Float[at.Array, "b 1 v"],
              prefix_kv_cache: _gemma.KVCache,
              prefix_mask: at.Bool[at.Array, "b p"],
              prefix_positions: at.Int[at.Array, "b p"],
              v1_len: at.Int[at.Array, ""],
              *,
              temprature: float = 0.,
              max_decoding_steps: int = 256,
              ) -> at.Int[at.Array, "b _s"]:
        """
        Output reasoning tokens after `prefill` using VLM0 (Expert2).

        Args:
            rng: PRNG key.
            last_logit: logit for the last token of the prefix,
                i.e., `<END_OF_PREFIX>` from VLM0.
            prefix_kv_cache: KV cache for the prefix.
                `<END_OF_PREFIX>` token included.
            prefix_mask: input mask for the prefix. Physical layout: [VLM1 | VLM0].
            prefix_positions: position id for the prefix.
            v1_len: number of VLM1 tokens (used to mask VLM1 from VLM0 decoded tokens).
            temprature: decoding temprature.
            max_decoding_steps: maximum decoding steps.

        """
        # prepare decoding
        step_rng = jax.random.fold_in(rng, 0)
        if temprature > 0.0:
            token = jax.random.categorical(step_rng,
                                           last_logit / temprature,
                                           axis=-1)
        else:
            token = jnp.argmax(last_logit, axis=-1)
        has_eos = jnp.any(
            token == PALIGEMMA_EOS_TOKEN,
            axis=1)
        all_eos = jnp.all(has_eos)
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps),
                                  dtype=token.dtype)

        # VLM0 decoded tokens should NOT attend to VLM1 KV cache
        # (VLM0 cannot see VLM1, consistent with _apply_stream_constraints in training)
        # prefix_mask physical layout: [VLM1(v1_len) | VLM0(v0_len)]
        # NOTE: v1_len is a traced (dynamic) value inside JIT, so we cannot use
        #       numpy-style slicing like prefix_mask[:, :v1_len].  Instead, build
        #       a boolean mask with jnp.arange comparison (same pattern as act()).
        col_indices = jnp.arange(prefix_mask.shape[1])
        vlm1_region = col_indices[None, :] < v1_len          # (1, p), broadcasts to (b, p)
        vlm0_only_prefix_mask = jnp.where(vlm1_region, False, prefix_mask)

        # Compute VLM0's last valid position for position tracking
        # VLM0 positions start at offset 0 (from seg_positions in prefill)
        vlm0_valid_count = jnp.sum(prefix_mask & ~vlm1_region, axis=1)  # [b]
        vlm0_last_pos = vlm0_valid_count - 1  # [b], last valid VLM0 position

        # hack to ensure the kv_cache's shape remains fixed throughout the while loop
        # KV cache is shape (l, b, t, k, h)
        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x,
                                ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0))),
                                prefix_kv_cache,)
        # attn_mask is shape (b, 1, prefix_len + 1 + max_decoding_steps)
        # Decoded tokens attend only to the VLM0 prefix via vlm0_only_prefix_mask.
        prefix_attn_mask = einops.repeat(vlm0_only_prefix_mask, "b p -> b s p", s=1)
        attn_mask = jnp.pad(prefix_attn_mask,
                            ((0, 0), (0, 0), (0, max_decoding_steps + 1)))
        # |<---------prefix-kv---------|<--------padded-kv------->|<--current-kv-->|
        # |<----------prefix---------->|<--max_decoding_steps---->|<------1------->|
        # |<-VLM1:False-><-VLM0:True-->|<---True--->|<---False--->|<-----True----->|
        attn_mask = attn_mask.at[:, :, -1].set(True)

        @at.typecheck
        def _wrap_cache(cache_appended: at.Float[at.Array, "l b t k h"],
                        step: at.Int[at.Array, ""],
                        ) -> at.Float[at.Array, "l b t-1 k h"]:
            new_value = cache_appended[:, :, -1]
            cache = cache_appended[:, :, :-1]
            cache = jax.lax.dynamic_update_index_in_dim(cache,
                                                        new_value,
                                                        prefix_mask.shape[1] + 1 + step,
                                                        axis=2)
            return cache
        
        def decode_step(carry):
            last_logit, output_tokens, kv_cache, attn_mask, _, step = carry

            step_rng = jax.random.fold_in(rng, step)
            # Sample token from last logit
            if temprature > 0.0:
                token = jax.random.categorical(step_rng,
                                               last_logit / temprature,
                                               axis=-1)
            else:
                token = jnp.argmax(last_logit, axis=-1)
            # force BOT at step 0
            token = jnp.where(
                step == 0,
                jnp.full_like(token, _tokenizer.BEGIN_OF_REASONING),
                token,
            )
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token)

            # Check for early stopping -->
            # stop if all batch elements have EOS token
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN,
                              axis=1)
            all_eos = jnp.all(has_eos)

            # Decode one step using VLM0 (Expert2)
            token_embedding = self.PaliGemma.llm(token, method="embed")
            # VLM0 positions continue from last valid VLM0 position
            positions = vlm0_last_pos[:, None] + step + 1
            (_, _, vlm0_pre_logit), kv_cache_appended = self.PaliGemma.llm(
                [None, None, token_embedding], mask=attn_mask, positions=positions, kv_cache=kv_cache
            )

            last_logit = self.PaliGemma.llm(vlm0_pre_logit, method="embedder_decode")
            kv_cache = jax.tree.map(
                lambda x: _wrap_cache(x, step),
                kv_cache_appended,
            )
            attn_mask = attn_mask.at[:, :, prefix_mask.shape[1] + 1 + step].set(True)
            return last_logit, output_tokens, kv_cache, attn_mask, all_eos, step + 1
        
        def decode_cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)
        
        _, suffix_txt_tokens, kv_cache, _, _, _ = \
            jax.lax.while_loop(
                decode_cond, decode_step,
                (last_logit, output_tokens, kv_cache, attn_mask, all_eos, 0),
                )
        
        return suffix_txt_tokens


    @at.typecheck
    def act(self,
            rng: at.KeyArrayLike,
            observation: _model.FuseObservation,
            prefix_cache: _gemma.KVCache,
            prefix_mask: at.Bool[at.Array, "b p"],
            prefix_positions: at.Int[at.Array, "b p"],
            v1_len: at.Int[at.Array, ""],  # Number of VLM1 tokens for masking VLM0 KV.
            *,
            num_steps: int | at.Int[at.Array, ""] = 10,
              ) -> at.Float[at.Array, "b ah ad"]:
        """
        Sample action after `prefill`.

        Args:
            rng: PRNG key.
            observation: FuseObservation.
            prefix_kv_cache: KV cache for the img + txt prefix.
                `<END_OF_PREFIX>` token included.
                `<BEGIN_OF_ACTION>` token not included.
            prefix_mask: input mask for the img + txt prefix.
            prefix_positions: position id for the img + txt prefix.
            num_steps: number of action denoising steps.
        
        Returns:
            at.Float[at.Array, "b ah ad"]: actions.
        """
        # first forward the <BEGIN_OF_ACTION> token
        boa_token = jnp.broadcast_to(
            _tokenizer.BEGIN_OF_ACTION, (prefix_mask.shape[0], 1))
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        boa_attn_mask = jnp.concatenate(
            [prefix_attn_mask, jnp.ones((prefix_attn_mask.shape[0], 1, 1), dtype=jnp.bool_)],
            axis=-1)
        # With seg_positions, positions[:, -1] is not necessarily the max.
        # (Physical order [VLM1, VLM0], while VLM0 positions start at 0.)
        # Training: action offset = v0_len + v1_len, BOA should be at that position.
        boa_positions = jnp.max(prefix_positions, axis=1, keepdims=True) + 1
        boa_token_embedding = self.PaliGemma.llm(boa_token, method="embed")
        (boa_pre_logit, _, _), img_txt_kv_cache = self.PaliGemma.llm(
            [boa_token_embedding, None, None], mask=boa_attn_mask, positions=boa_positions, kv_cache=prefix_cache
        )
        img_txt_mask = jnp.pad(prefix_mask, ((0, 0), (0, 1)),
                                 constant_values=1)

        # Action-specific prefix mask hides the VLM0 KV cache.
        # img_txt_mask layout: [VLM1(v1_len) | VLM0(v0_len) | BOA(1)].
        # Training forbids Action→VLM0 via _apply_stream_constraints; keep the same at inference.
        prefix_indices = jnp.arange(img_txt_mask.shape[1])
        is_vlm0_region = (prefix_indices >= v1_len) & (prefix_indices < img_txt_mask.shape[1] - 1)
        action_prefix_visible = img_txt_mask & ~is_vlm0_region[None, :]  # VLM1 + BOA visible, VLM0 masked.

        # Suffix start position = max_prefix_pos + 2
        # (BOA takes +1, suffix starts at +2).
        suffix_start_pos = jnp.max(prefix_positions, axis=1)[:, None] + 2  # [b, 1]

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def denoise_step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_state_action(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(action_prefix_visible, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            # Suffix positions use suffix_start_pos rather than sum(img_txt_mask).
            # Under seg_positions, token count != max position; continue from max_prefix_pos.
            positions = suffix_start_pos + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out, _), _ = self.PaliGemma.llm(
                [None, suffix_tokens, None], mask=full_attn_mask, positions=positions, kv_cache=img_txt_kv_cache,
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def denoise_cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(denoise_cond, denoise_step, (noise, 1.0))
        return x_0

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        """
        Sample actions and calculate the text loss for validation.

        Args:
            rng: PRNG key.
            observation: FuseObservation.
            num_steps: number of action denoising steps.
        
        Returns:
            tuple: A tuple containing
                - actions: actions.
                - info: dict[str, at.Array].
                    - text_loss: CE loss for the text suffix.
                    - num_action_loss_fraction: fraction of samples used
                        for action loss calculation.
        """

        observation = _model.preprocess_observation(None, observation, train=False, image_keys=list(observation.images.keys()))

        info = {}
        info['num_action_loss_fraction'] = (
            jnp.sum(observation.diffusion_loss_mask) /
            observation.diffusion_loss_mask.shape[0]
        )

        # Prefix: VLM1 (images) + VLM0 (ges_images + hand + text), no action tokens yet.
        vlm1_tokens, vlm1_mask, vlm1_ar, vlm0_tokens, vlm0_mask, vlm0_ar = self.embed_img_txt_dual(observation)

        prefix_mask = jnp.concatenate([vlm1_mask, vlm0_mask], axis=1)
        prefix_ar_mask = jnp.concatenate([vlm1_ar, vlm0_ar], axis=1)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        (vlm1_pre_logits, _, vlm0_pre_logits), kv_cache = self.PaliGemma.llm(
            [vlm1_tokens, None, vlm0_tokens], mask=prefix_attn_mask, positions=prefix_positions
        )

        # text ce loss (from VLM0 / expert 2)
        txt_targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            _gemma.PALIGEMMA_VOCAB_SIZE,
        )
        txt_logits = self.PaliGemma.llm(
            vlm0_pre_logits[:, -1-txt_targets.shape[1]: -1],
            method="embedder_decode",
        )
        txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)
        txt_loss_mask = observation.token_loss_mask[:, 1:]
        txt_token_pplx = jnp.sum(txt_targets * txt_logp, axis=-1)
        txt_loss = (
            -jnp.sum(txt_token_pplx * txt_loss_mask, axis=-1) /
            jnp.clip(jnp.sum(txt_loss_mask, axis=-1), 1)
            )

        info['text_loss'] = txt_loss

        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_state_action(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_s, suffix_attn_mask], axis=-1)
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out, _), _ = self.PaliGemma.llm(
                [None, suffix_tokens, None], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0, info