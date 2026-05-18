import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
import openpi.models.tokenizer as _tokenizer
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def _as_bt12(x, b):
    """Convert gesture data to [b, t, 12]; return zeros if None."""
    if x is None:
        logger.warning("hand_pose is None, returning zero tensor with shape [b, 16, 12]")
        return jnp.zeros((b, 16, 12), dtype=jnp.float32)

    if hasattr(x, "shape") and len(x.shape) >= 2:
        return x

    return jnp.zeros((b, 8, 12), dtype=jnp.float32)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way mask_ar can be used to setup several
    types of attention, e.g.:

      [[1 1 1 1 1 1]]: pure causal attention.
      [[0 0 0 1 1 1]]: prefix-lm attention.
      [[1 0 1 0 1 0]]: causal attention between blocks.

    Args:
      input_mask: bool[B, N] true if part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask.
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


@dataclasses.dataclass(frozen=True)
class ReasoningConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 150

    diffusion_loss_coeff: float = 1.0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.REASONING

    @override
    def create(self, rng: at.KeyArrayLike) -> "ReasoningModel":
        return ReasoningModel(self, rngs=nnx.Rngs(rng))

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
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter)
            has_lora = True

        if has_lora:
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")))
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class ReasoningModel(_model.BaseModel):
    """Lightweight reasoning-only model: PaliGemma LLM + hand_pose MLP, no action diffusion."""

    def __init__(self, config: ReasoningConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config],
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

        self.hand_pose_mlp_1 = nnx.Linear(12, 256, rngs=rngs)
        self.hand_pose_mlp_2 = nnx.Linear(256, paligemma_config.width, rngs=rngs)

        self.diffusion_loss_coeff = config.diffusion_loss_coeff

    @at.typecheck
    def embed_img_txt(
        self, obs: _model.FuseObservation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        """Embed images, hand pose, and tokenized text."""
        input_mask = []
        ar_mask = []
        embeddings = []

        # Image embeddings.
        for name in obs.images:
            image_emb, _ = self.PaliGemma.img(obs.images[name], train=False)
            embeddings.append(image_emb)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_emb.shape[1])
            )
            ar_mask.append(jnp.zeros_like(input_mask[-1], dtype=jnp.int32))

        # Hand pose sequence: inserted between image and text.
        b = obs.tokenized_prompt.shape[0]
        hand_pose_seq = getattr(obs, "hand_pose", None)
        hand_pose_mask = getattr(obs, "hand_pose_mask", None)

        if hand_pose_seq is not None and hand_pose_mask is not None:
            hand_pose_seq = _as_bt12(hand_pose_seq, b)  # [b, t, 12]
            hp_mask = hand_pose_mask.astype(jnp.bool_)

            hp_emb = self.hand_pose_mlp_1(hand_pose_seq)  # [b, t, 256]
            hp_emb = nnx.swish(hp_emb)
            hp_emb = self.hand_pose_mlp_2(hp_emb)  # [b, t, emb]

            embeddings.append(hp_emb)
            input_mask.append(hp_mask)
            ar_mask.append(jnp.zeros_like(hp_mask, dtype=jnp.int32))

        # Text embeddings.
        assert obs.tokenized_prompt is not None, "Tokenized prompt is required"
        assert obs.tokenized_prompt_mask is not None, "Tokenized prompt mask is required"
        assert obs.token_ar_mask is not None, "Token ar mask is required"
        assert obs.token_loss_mask is not None, "Token loss mask is required"

        txt_emb = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        embeddings.append(txt_emb)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask.append(obs.token_ar_mask)

        embeddings = jnp.concatenate(embeddings, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1)
        return embeddings, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.FuseObservation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b"], dict[str, at.Array]]:
        preprocess_rng = rng  # unused, kept for API compatibility.
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train,
            image_keys=list(observation.images.keys()),
            tokenizer=_tokenizer.FusePaligemmaTokenizer,
        )

        img_txt_tokens, img_txt_mask, img_txt_ar_mask = self.embed_img_txt(observation)
        attn_mask = make_attn_mask(img_txt_mask, img_txt_ar_mask)
        positions = jnp.cumsum(img_txt_mask, axis=1) - 1
        (img_txt_pre_logits,), _ = self.PaliGemma.llm(
            [img_txt_tokens], mask=attn_mask, positions=positions
        )

        # Text cross-entropy loss.
        txt_targets = jax.nn.one_hot(
            observation.tokenized_prompt[:, 1:],
            _gemma.PALIGEMMA_VOCAB_SIZE,
        )
        txt_logits = self.PaliGemma.llm(
            img_txt_pre_logits[:, -1 - txt_targets.shape[1] : -1],
            method="embedder_decode",
        )
        txt_logp = jax.nn.log_softmax(txt_logits, axis=-1)
        txt_loss_mask = observation.token_loss_mask[:, 1:]

        txt_token_pplx = jnp.sum(txt_targets * txt_logp, axis=-1)
        txt_loss = (
            -jnp.sum(txt_token_pplx * txt_loss_mask, axis=-1)
            / jnp.clip(jnp.sum(txt_loss_mask, axis=-1), 1)
        )

        return txt_loss, {"text_loss": txt_loss}

    @at.typecheck
    def prefill(
        self,
        rng: at.KeyArrayLike,
        observation: _model.FuseObservation,
        *,
        temprature: float = 0.0,
    ) -> tuple[
        _model.FuseObservation,
        _gemma.KVCache,
        at.Int[at.Array, "b 1"],
        at.Float[at.Array, "b 1 v"],
        at.Bool[at.Array, "b s"],
        at.Int[at.Array, "b s"],
        at.Bool[at.Array, "b"],
    ]:
        """Prefill the KV cache with the prefix. Used for policy serving."""
        observation = _model.preprocess_observation(
            None, observation, train=False, image_keys=list(observation.images.keys())
        )
        first_one_indices = jnp.argmax(observation.token_ar_mask, axis=-1)
        padding_mask = jnp.arange(observation.token_ar_mask.shape[-1]) >= first_one_indices[..., jnp.newaxis]
        observation = dataclasses.replace(
            observation,
            tokenized_prompt=jnp.where(padding_mask, 0, observation.tokenized_prompt),
            tokenized_prompt_mask=jnp.logical_not(padding_mask),
        )

        prefix_token_embeddings, prefix_mask, prefix_ar_mask = self.embed_img_txt(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (pre_logit,), kv_cache = self.PaliGemma.llm(
            [prefix_token_embeddings], mask=prefix_attn_mask, positions=prefix_positions
        )
        eop_indices = prefix_positions[:, -1]
        eop_pre_logit = jnp.take_along_axis(pre_logit, eop_indices[:, None, None], axis=1)
        eop_logit = self.PaliGemma.llm(eop_pre_logit, method="embedder_decode")

        valid_tokens = jnp.array([_tokenizer.BEGIN_OF_ACTION, _tokenizer.BEGIN_OF_REASONING])
        valid_mask = jnp.full((1, 1, eop_logit.shape[-1]), -jnp.inf)
        valid_mask = valid_mask.at[:, :, valid_tokens].set(0)
        eop_logit = eop_logit + valid_mask
        if temprature > 0.0:
            token = jax.random.categorical(rng, eop_logit / temprature, axis=-1)
        else:
            token = jnp.argmax(eop_logit, axis=-1)
        has_boa = jnp.any(token == _tokenizer.BEGIN_OF_ACTION, axis=1)

        return observation, kv_cache, token, eop_logit, prefix_mask, prefix_positions, has_boa

    @at.typecheck
    def reason(
        self,
        rng: at.KeyArrayLike,
        last_logit: at.Float[at.Array, "b 1 v"],
        prefix_kv_cache: _gemma.KVCache,
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_positions: at.Int[at.Array, "b p"],
        *,
        temprature: float = 0.0,
        max_decoding_steps: int = 256,
    ) -> at.Int[at.Array, "b _s"]:
        """Autoregressive decoding after prefill."""
        step_rng = jax.random.fold_in(rng, 0)
        if temprature > 0.0:
            token = jax.random.categorical(step_rng, last_logit / temprature, axis=-1)
        else:
            token = jnp.argmax(last_logit, axis=-1)
        has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=1)
        all_eos = jnp.all(has_eos)
        output_tokens = jnp.zeros((last_logit.shape[0], max_decoding_steps), dtype=token.dtype)

        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, max_decoding_steps), (0, 0), (0, 0))),
            prefix_kv_cache,
        )
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=1)
        attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_decoding_steps + 1)))
        attn_mask = attn_mask.at[:, :, -1].set(True)

        @at.typecheck
        def _wrap_cache(
            cache_appended: at.Float[at.Array, "l b t k h"],
            step: at.Int[at.Array, ""],
        ) -> at.Float[at.Array, "l b t-1 k h"]:
            new_value = cache_appended[:, :, -1]
            cache = cache_appended[:, :, :-1]
            cache = jax.lax.dynamic_update_index_in_dim(cache, new_value, prefix_mask.shape[1] + 1 + step, axis=2)
            return cache

        def decode_step(carry):
            last_logit, output_tokens, kv_cache, attn_mask, _, step = carry
            step_rng = jax.random.fold_in(rng, step)
            if temprature > 0.0:
                token = jax.random.categorical(step_rng, last_logit / temprature, axis=-1)
            else:
                token = jnp.argmax(last_logit, axis=-1)
            token = jnp.where(step == 0, jnp.full_like(token, _tokenizer.BEGIN_OF_REASONING), token)
            output_tokens = put_along_last_axis(
                output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token
            )
            has_eos = jnp.any(token == PALIGEMMA_EOS_TOKEN, axis=1)
            all_eos = jnp.all(has_eos)

            token_embedding = self.PaliGemma.llm(token, method="embed")
            positions = prefix_positions[:, [-1]] + step + 1
            (last_pre_logit,), kv_cache_appended = self.PaliGemma.llm(
                [token_embedding], mask=attn_mask, positions=positions, kv_cache=kv_cache
            )
            last_logit = self.PaliGemma.llm(last_pre_logit, method="embedder_decode")
            kv_cache = jax.tree.map(lambda x: _wrap_cache(x, step), kv_cache_appended)
            attn_mask = attn_mask.at[:, :, prefix_mask.shape[1] + 1 + step].set(True)
            return last_logit, output_tokens, kv_cache, attn_mask, all_eos, step + 1

        def decode_cond(carry):
            _, _, _, _, all_eos, step = carry
            return (~all_eos) & (step < max_decoding_steps)

        _, suffix_txt_tokens, _, _, _, _ = jax.lax.while_loop(
            decode_cond, decode_step, (last_logit, output_tokens, kv_cache, attn_mask, all_eos, 0)
        )
        return suffix_txt_tokens

