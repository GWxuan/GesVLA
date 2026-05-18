import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from openpi.shared import image_tools
import openpi.shared.array_typing as at
import openpi.transforms_2vlm as _transforms

logger = logging.getLogger("openpi")

ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0_GES = "pi0_ges"



# The model always expects these images
IMAGE_KEYS = (
    "0_rgb", "1_rgb", "2_rgb", 
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model."""
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]

    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    origin_episode_idx: at.Int[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            origin_episode_idx=data.get("origin_episode_idx"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


@at.typecheck
@struct.dataclass
class FuseObservation(Observation):
    """Observation for the pi0_fuse / pi0_ges model."""

    diffusion_loss_mask: at.Bool[ArrayT, "*b"] | None = None

    # hand pose sequence
    hand_pose: at.Array | None = None
    hand_pose_mask: at.Array | None = None

    # NEW: gesture images (for VLM0)
    ges_images: dict[str, at.Float[ArrayT, "*b h w c"]] | None = None
    ges_image_masks: dict[str, at.Bool[ArrayT, "*b"]] | None = None

    origin_episode_idx: at.Int[ArrayT, "*b"] | None = None
    idx: at.Int[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "FuseObservation[ArrayT]":
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        # main images uint8 -> [-1,1]
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0

        # NEW: ges images uint8 -> [-1,1]
        if "ges_images" in data and data["ges_images"] is not None:
            for key in data["ges_images"]:
                if data["ges_images"][key].dtype == np.uint8:
                    data["ges_images"][key] = data["ges_images"][key].astype(np.float32) / 255.0 * 2.0 - 1.0

        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),

            diffusion_loss_mask=data.get("diffusion_loss_mask"),
            hand_pose=data.get("hand_pose"),
            hand_pose_mask=data.get("hand_pose_mask"),

            # NEW
            ges_images=data.get("ges_images"),
            ges_image_masks=data.get("ges_image_masks"),

            origin_episode_idx=data.get("origin_episode_idx"),
            idx=data.get("idx"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")

        # NEW
        result["ges_images"] = result.pop("ges_images")
        result["ges_image_masks"] = result.pop("ges_image_masks")

        return result
    
# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]

def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation | FuseObservation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    allow_geometric_augmentation: bool = False,
    tokenizer: any = None,
) -> Observation | FuseObservation:
    """Preprocess observations: resize/augment images and fill default masks."""

    # ---- main images must contain image_keys ----
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    def _proc_image_dict(imgs: dict[str, ArrayT], masks: dict[str, ArrayT] | None):
        out_imgs = {}
        out_msks = {}

        for key, image in imgs.items():
            if image.shape[1:3] != image_resolution:
                logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
                image = image_tools.resize_with_pad(image, *image_resolution)

            if train:
                # [-1,1] -> [0,1]
                image01 = image / 2.0 + 0.5

                transforms = [
                    augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ]
                sub_rngs = jax.random.split(rng, image01.shape[0]) if rng is not None else None
                if sub_rngs is None:
                    # if rng is None but train=True (should not happen), skip aug
                    image01 = image01
                else:
                    image01 = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image01)

                # [0,1] -> [-1,1]
                image = image01 * 2.0 - 1.0

            out_imgs[key] = image

        for key in out_imgs:
            if masks is None or key not in masks:
                out_msks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
            else:
                out_msks[key] = jnp.asarray(masks[key])

        return out_imgs, out_msks

    # main images
    out_images, out_masks = _proc_image_dict(observation.images, observation.image_masks)

    if not isinstance(observation, FuseObservation):
        return Observation(
            images=out_images,
            image_masks=out_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

    # FuseObservation: also process ges_images if present
    ges_out_images = None
    ges_out_masks = None
    if observation.ges_images is not None:
        ges_out_images, ges_out_masks = _proc_image_dict(observation.ges_images, observation.ges_image_masks)

    return FuseObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        diffusion_loss_mask=observation.diffusion_loss_mask,
        hand_pose=observation.hand_pose,
        hand_pose_mask=observation.hand_pose_mask,

        # NEW
        ges_images=ges_out_images,
        ges_image_masks=ges_out_masks,

        origin_episode_idx=observation.origin_episode_idx,
        idx=observation.idx,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation) -> tuple[Actions, dict[str, at.Array]]: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Model params not found at: {params_path}")

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
