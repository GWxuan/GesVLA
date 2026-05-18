from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
import torch
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")

from openpi.timer_utils import Timer, timed_function


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """



@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)




@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        return (x - stats.mean) / (stats.std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        return x * (stats.std + 1e-6) + stats.mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    COORD_PATTERN = re.compile(r'\(([0-9.]+),\s*([0-9.]+)\)')
    reference_frame: str = "0_rgb"

    @timed_function("ResizeImages")
    def __call__(self, data: DataDict) -> DataDict:
        resized = {}
        reference_params = None
        for k, v in data["image"].items():
            v = np.asarray(v)
            if v.dtype != np.uint8:
                v = v.astype(np.uint8)
            if v.shape[0] != self.height or v.shape[1] != self.width:
                resized[k] = image_tools.resize_with_pad(v, self.height, self.width)
                if k == self.reference_frame:
                    original_height, original_width = v.shape[:2]
                    reference_params = self._calculate_transform_params(
                        original_width, original_height, self.width, self.height
                    )
            else:
                resized[k] = v

        data["image"] = resized

        if "ges_images" in data:
            resized_ges = {}
            reference_params = None
            for k, v in data["ges_images"].items():
                v = np.asarray(v)
                if v.dtype != np.uint8:
                    v = v.astype(np.uint8)
                if v.shape[0] != self.height or v.shape[1] != self.width:
                    resized_ges[k] = image_tools.resize_with_pad(v, self.height, self.width)
                    if k == "ges_0_rgb":
                        original_height, original_width = v.shape[:2]
                        reference_params = self._calculate_transform_params(
                            original_width, original_height, self.width, self.height
                        )
                else:
                    resized_ges[k] = v
                data["ges_images"] = resized_ges

        if "thought" in data and reference_params is not None:
            data = self._transform_coordinates_in_thought(data, reference_params)
        if "hand_pose" in data and reference_params is not None:
            data = self._transform_hand_pose(data, reference_params)
        return data
    def _calculate_transform_params(self, orig_w: int, orig_h: int, target_w: int, target_h: int) -> dict:
        """Calculate image transformation parameters (scale, padding offsets)."""
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        
        return {
            'scale': scale,
            'new_width': new_w,
            'new_height': new_h,
            'pad_left': pad_left,
            'pad_top': pad_top,
            'pad_right': pad_w - pad_left,
            'pad_bottom': pad_h - pad_top,
            'original_width': orig_w,
            'original_height': orig_h,
            'target_width': target_w,
            'target_height': target_h
        }
    
    def _transform_coordinates_in_thought(self, data: DataDict, transform_params: dict) -> DataDict:
        transformed_thoughts = []
        
        for thought_text in data["thought"]:
            if isinstance(thought_text, str):
                transformed_text = self._transform_coordinates_in_text(thought_text, transform_params)
                transformed_thoughts.append(transformed_text)
            else:
                transformed_thoughts.append(thought_text)
        
        data["thought"] = transformed_thoughts
        return data
    
    def _transform_coordinates_in_text(self, text: str, transform_params: dict) -> str:
        """Transform coordinates in a single text string."""
        
        def replace_coordinate(match):
            x_orig = float(match.group(1))
            y_orig = float(match.group(2))
            
            x_new, y_new = self._transform_single_coordinate(
                x_orig, y_orig, transform_params
            )
            
            return f"({x_new:.4f}, {y_new:.4f})"
        
        # Replace all coordinate patterns using regex.
        return self.COORD_PATTERN.sub(replace_coordinate, text)
    
    def _transform_single_coordinate(self, x_orig: float, y_orig: float, params: dict) -> tuple[float, float]:
        """Transform a single coordinate pair through resize+pad mapping."""
        # Pixel coordinates in original image space
        x_pixel = x_orig * params['original_width']
        y_pixel = y_orig * params['original_height']
        
        # Scaled coordinates
        x_scaled = x_pixel * params['scale']
        y_scaled = y_pixel * params['scale']
        
        # Add padding offset
        x_padded = x_scaled + params['pad_left']
        y_padded = y_scaled + params['pad_top']
        
        # Normalize to new image dimensions
        x_new = x_padded / params['target_width']
        y_new = y_padded / params['target_height']
        
        return x_new, y_new
    def _transform_hand_pose(self, data: DataDict, transform_params: dict) -> DataDict:
        """Transform hand pose coordinates through the resize+pad mapping."""
        hand_pose = data["hand_pose"]
        if isinstance(hand_pose, np.ndarray):
            hand_pose = hand_pose.copy()
        elif isinstance(hand_pose, torch.Tensor):
            hand_pose = hand_pose.clone()
        else:
            logging.warning("Unknown hand pose type: %s", type(hand_pose))
            return data
        
        
        if len(hand_pose.shape) == 2:  # [t, 12] multiple timesteps
            for t in range(hand_pose.shape[0]):
                if torch.allclose(hand_pose[t], torch.tensor(0.0), atol=1e-8):
                    continue
            
                # Process 4 keypoints, each with 3 coordinates (x, y, z)
                for i in range(4):
                    x_idx = i * 3
                    y_idx = i * 3 + 1
                    z_idx = i * 3 + 2
                    
                    x_orig = hand_pose[t, x_idx]
                    y_orig = hand_pose[t, y_idx]
                    z = hand_pose[t, z_idx]
                    

                    x_new, y_new = self._transform_single_coordinate(x_orig, y_orig, transform_params)
                    
                    # Update transformed coordinates; z remains unchanged
                    hand_pose[t, x_idx] = x_new
                    hand_pose[t, y_idx] = y_new
        #print(f"hand_pose change to: {hand_pose.numpy()}")
                    
        
        data["hand_pose"] = hand_pose
        return data



@dataclasses.dataclass(frozen=True)
class GesTokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.FusePaligemmaTokenizer

    @timed_function("GesTokenizePrompt")
    def __call__(self, data: DataDict) -> DataDict:
        thought = data.pop("thought", None)
        if thought is None:
            raise ValueError("Thought is required")
        (
            tokens,
            token_mask,
            ar_mask,
            text_loss_mask,
            diffusion_loss_mask,
        ) = self.tokenizer.tokenize(
            thought,
            )
        
        

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": text_loss_mask,
            "diffusion_loss_mask": diffusion_loss_mask,
            }


@dataclasses.dataclass(frozen=True)
class ExtractThoughts(DataTransformFn):
    tokenizer: _tokenizer.FusePaligemmaTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if "tokenized_suffix" not in data:
            return data
        tokens = data["tokenized_suffix"]
        thoughts = self.tokenizer.extract_thoughts(tokens)
        return {**data, "thoughts": thoughts}


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


@dataclasses.dataclass(frozen=True)
class GeometricAugmentation(DataTransformFn):
    """Geometric data augmentation (random crop and translation)."""
    apply_augmentation: bool = False

    @timed_function("GeometricAugmentation")
    def __call__(self, data: DataDict) -> DataDict:
        if not self.apply_augmentation:
            return data
            
        import random
        
        aug_type = random.randint(0, 1)
        
        if aug_type == 0:
            crop_transform = RandomCropWithCoordSync(
                height=224,
                width=224,
                scale_range=(0.9, 1.0),
            )
            augmented_data = crop_transform(data)
        else:
            translate_transform = RandomTranslateWithCoordSync(
                max_translate=20,
            )
            augmented_data = translate_transform(data)
        
        return augmented_data


@dataclasses.dataclass(frozen=True)
class RandomCropWithCoordSync(DataTransformFn):
    """Random crop with synchronized coordinate transformation."""
    height: int
    width: int
    scale_range: tuple[float, float] = (0.9, 1.0)

    @timed_function("RandomCropWithCoordSync")
    def __call__(self, data: DataDict) -> DataDict:
        # Get original image dimensions.
        first_key = next(iter(data["image"].keys()))
        original_img = data["image"][first_key]
        orig_h, orig_w = original_img.shape[:2]
        
        # Compute crop dimensions.
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        crop_h = int(orig_h * scale)
        crop_w = int(orig_w * scale)
        
        # Randomly select crop start position.
        start_y = np.random.randint(0, orig_h - crop_h + 1)
        start_x = np.random.randint(0, orig_w - crop_w + 1)
        
        # Crop and resize images.
        cropped_images = {}
        for key, img in data["image"].items():
            if "wrist" in key:
                cropped_images[key] = img
                continue
            if len(data["image"]) == 4:
                # Re-randomize per view when there are 4 images.
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
                crop_h = int(orig_h * scale)
                crop_w = int(orig_w * scale)
                start_y = np.random.randint(0, orig_h - crop_h + 1)
                start_x = np.random.randint(0, orig_w - crop_w + 1)
            cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
            resized = image_tools.resize_with_pad(cropped, self.height, self.width)
            cropped_images[key] = resized
        data["image"] = cropped_images
        
        # Compute transformation parameters for coordinate sync.
        transform_params = {
            'crop_start_x': start_x,
            'crop_start_y': start_y,
            'crop_width': crop_w,
            'crop_height': crop_h,
            'original_width': orig_w,
            'original_height': orig_h,
            'target_width': self.width,
            'target_height': self.height
        }
        
        # Transform thought coordinates.
        if "thought" in data:
            data = self._transform_thought_coordinates(data, transform_params)
        
        # Transform hand_pose coordinates.
        if "hand_pose" in data:
            data = self._transform_hand_pose_for_crop(data, transform_params)

        return data
    
    def _transform_thought_coordinates(self, data: DataDict, transform_params: dict) -> DataDict:
        """Transform coordinates embedded in thought text strings."""
        transformed_thoughts = []
        
        for thought_text in data["thought"]:
            if isinstance(thought_text, str):           
                transformed_text = self._transform_coordinates_in_text(thought_text, transform_params)
                transformed_thoughts.append(transformed_text)
            else:
                transformed_thoughts.append(thought_text)
        
        data["thought"] = transformed_thoughts
        return data
    
    def _transform_coordinates_in_text(self, text: str, transform_params: dict) -> str:
        """Transform coordinates in text for crop augmentation."""
        coord_pattern = re.compile(r'\(([0-9.]+),\s*([0-9.]+)\)')
        
        def replace_coordinate(match):
            x_orig = float(match.group(1))
            y_orig = float(match.group(2))
            
            x_new, y_new = self._transform_single_coordinate_for_crop(
                x_orig, y_orig, transform_params
            )
            
            return f"({x_new:.4f}, {y_new:.4f})"
        
        return coord_pattern.sub(replace_coordinate, text)
    
    def _transform_single_coordinate_for_crop(self, x_orig: float, y_orig: float, params: dict) -> tuple[float, float]:
        """Transform a single coordinate pair for crop augmentation."""
        # Pixel coordinates in original image space
        x_pixel = x_orig * params['original_width']
        y_pixel = y_orig * params['original_height']
        
        # Subtract crop start position
        x_cropped = x_pixel - params['crop_start_x']
        y_cropped = y_pixel - params['crop_start_y']
        
        # Normalize to cropped region
        x_normalized = x_cropped / params['crop_width']
        y_normalized = y_cropped / params['crop_height']
        
        # Already normalized to [0, 1] range.
        return x_normalized, y_normalized
    
    def _transform_hand_pose_for_crop(self, data: DataDict, transform_params: dict) -> DataDict:
        """Transform hand pose coordinates for crop augmentation."""
        hand_pose = data["hand_pose"]
        
        if isinstance(hand_pose, np.ndarray):
            hand_pose = hand_pose.copy()
        elif isinstance(hand_pose, torch.Tensor):
            hand_pose = hand_pose.clone()
        else:
            return data
        
        if len(hand_pose.shape) == 1:  # [12] single timestep
            hand_pose = hand_pose.reshape(1, -1)
        
        if len(hand_pose.shape) == 2:  # [t, 12] multiple timesteps
            for t in range(hand_pose.shape[0]):
                if self._is_zero_padding(hand_pose[t]):
                    continue
                
                # Process 4 keypoints
                for i in range(4):
                    x_idx = i * 3
                    y_idx = i * 3 + 1
                    z_idx = i * 3 + 2
                    
                    x_orig = hand_pose[t, x_idx]
                    y_orig = hand_pose[t, y_idx]
                    
                    # Transform x, y coordinates.
                    x_new, y_new = self._transform_single_coordinate_for_crop(
                        x_orig, y_orig, transform_params
                    )
                    
                    hand_pose[t, x_idx] = x_new
                    hand_pose[t, y_idx] = y_new
        
        data["hand_pose"] = hand_pose
        return data
    
    def _is_zero_padding(self, hand_pose_frame: np.ndarray | torch.Tensor) -> bool:
        """Check whether the hand pose frame is all-zero padding."""
        if isinstance(hand_pose_frame, np.ndarray):
            return np.allclose(hand_pose_frame, 0.0, atol=1e-8)
        elif isinstance(hand_pose_frame, torch.Tensor):
            return torch.allclose(hand_pose_frame, torch.tensor(0.0), atol=1e-8)
        else:
            return np.allclose(np.asarray(hand_pose_frame), 0.0, atol=1e-8)


@dataclasses.dataclass(frozen=True)
class RandomTranslateWithCoordSync(DataTransformFn):
    """Random translation with synchronized coordinate transformation."""
    max_translate: int = 20

    @timed_function("RandomTranslateWithCoordSync")
    def __call__(self, data: DataDict) -> DataDict:
        # Generate random translation offsets.
        tx = np.random.randint(-self.max_translate, self.max_translate + 1)
        ty = np.random.randint(-self.max_translate, self.max_translate + 1)
        
        if tx == 0 and ty == 0:
            return data

        # Translate images.
        translated_images = {}
        for key, img in data["image"].items():
            if "wrist" in key:
                translated_images[key] = img
                continue
            if len(data["image"]) == 4:
                # Re-randomize per view when there are 4 images.
                tx = np.random.randint(-self.max_translate, self.max_translate + 1)
                ty = np.random.randint(-self.max_translate, self.max_translate + 1)
            h, w = img.shape[:2]
            translated = self._translate_image(img, tx, ty)
            translated_images[key] = translated
        
        data["image"] = translated_images
        
        # Transform parameters for coordinate sync.
        transform_params = {
            'translate_x': tx,
            'translate_y': ty,
            'image_width': w,
            'image_height': h
        }

        # Note: thought/hand_pose coordinate transforms for translation are
        # currently disabled pending further validation.
        
        return data
    
    def _translate_image(self, img: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """Apply pixel-level translation to an image with zero-padding."""
        h, w = img.shape[:2]
        translated = np.zeros_like(img)
        
        # Compute valid source and destination regions for the translation.
        if tx >= 0 and ty >= 0:
            # Translate right-down
            src_x1, src_y1 = 0, 0
            src_x2, src_y2 = w - tx, h - ty
            dst_x1, dst_y1 = tx, ty
            dst_x2, dst_y2 = w, h
        elif tx >= 0 and ty < 0:
            # Translate right-up
            src_x1, src_y1 = 0, -ty
            src_x2, src_y2 = w - tx, h
            dst_x1, dst_y1 = tx, 0
            dst_x2, dst_y2 = w, h + ty
        elif tx < 0 and ty >= 0:
            # Translate left-down
            src_x1, src_y1 = -tx, 0
            src_x2, src_y2 = w, h - ty
            dst_x1, dst_y1 = 0, ty
            dst_x2, dst_y2 = w + tx, h
        else:  # tx < 0 and ty < 0
            # Translate left-up
            src_x1, src_y1 = -tx, -ty
            src_x2, src_y2 = w, h
            dst_x1, dst_y1 = 0, 0
            dst_x2, dst_y2 = w + tx, h + ty
        
        # Ensure coordinates are within valid bounds
        src_x1, src_y1 = max(0, src_x1), max(0, src_y1)
        src_x2, src_y2 = min(w, src_x2), min(h, src_y2)
        dst_x1, dst_y1 = max(0, dst_x1), max(0, dst_y1)
        dst_x2, dst_y2 = min(w, dst_x2), min(h, dst_y2)
        
        # Copy image data.
        if src_x2 > src_x1 and src_y2 > src_y1:
            translated[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        
        return translated
        
    def _transform_thought_coordinates_for_translate(self, data: DataDict, transform_params: dict) -> DataDict:
        """Transform thought coordinates for translation augmentation."""
        transformed_thoughts = []
        
        for thought_text in data["thought"]:
            if isinstance(thought_text, str):
                
                # Transform coordinates.
                transformed_text = self._transform_coordinates_in_text_for_translate(thought_text, transform_params)
                transformed_thoughts.append(transformed_text)
            else:
                transformed_thoughts.append(thought_text)
        
        data["thought"] = transformed_thoughts
        return data
    
    def _transform_coordinates_in_text_for_translate(self, text: str, transform_params: dict) -> str:
        """Transform text coordinates for translation augmentation."""
        coord_pattern = re.compile(r'\(([0-9.]+),\s*([0-9.]+)\)')
        
        def replace_coordinate(match):
            x_orig = float(match.group(1))
            y_orig = float(match.group(2))
            
            # Transform coordinates.
            x_new, y_new = self._transform_single_coordinate_for_translate(
                x_orig, y_orig, transform_params
            )
            
            return f"({x_new:.4f}, {y_new:.4f})"
        
        return coord_pattern.sub(replace_coordinate, text)
    
    def _transform_single_coordinate_for_translate(self, x_orig: float, y_orig: float, params: dict) -> tuple[float, float]:
        """Transform a single coordinate pair for translation augmentation."""
        # Pixel coordinates in original image space
        x_pixel = x_orig * params['image_width']
        y_pixel = y_orig * params['image_height']
        
        # Apply translation offset
        x_translated = x_pixel + params['translate_x']
        y_translated = y_pixel + params['translate_y']
        
        # Normalize back to [0, 1] range
        x_normalized = x_translated / params['image_width']
        y_normalized = y_translated / params['image_height']
        
        return x_normalized, y_normalized
    
    def _transform_hand_pose_for_translate(self, data: DataDict, transform_params: dict) -> DataDict:
        """Transform hand pose coordinates for translation augmentation."""
        hand_pose = data["hand_pose"]
        
        if isinstance(hand_pose, np.ndarray):
            hand_pose = hand_pose.copy()
        elif isinstance(hand_pose, torch.Tensor):
            hand_pose = hand_pose.clone()
        else:
            return data
        
        if len(hand_pose.shape) == 1:
            hand_pose = hand_pose.reshape(1, -1)
        
        if len(hand_pose.shape) == 2:
            for t in range(hand_pose.shape[0]):
                if self._is_zero_padding(hand_pose[t]):
                    continue
                
                # Process 4 keypoints
                for i in range(4):
                    x_idx = i * 3
                    y_idx = i * 3 + 1
                    z_idx = i * 3 + 2
                    
                    x_orig = hand_pose[t, x_idx]
                    y_orig = hand_pose[t, y_idx]
                    
                    # Transform x, y coordinates.
                    x_new, y_new = self._transform_single_coordinate_for_translate(
                        x_orig, y_orig, transform_params
                    )
                    
                    hand_pose[t, x_idx] = x_new
                    hand_pose[t, y_idx] = y_new
        
        data["hand_pose"] = hand_pose
        return data
    
    def _is_zero_padding(self, hand_pose_frame: np.ndarray | torch.Tensor) -> bool:
        """Check whether the hand pose frame is all-zero padding."""
        if isinstance(hand_pose_frame, np.ndarray):
            return np.allclose(hand_pose_frame, 0.0, atol=1e-8)
        elif isinstance(hand_pose_frame, torch.Tensor):
            return torch.allclose(hand_pose_frame, torch.tensor(0.0), atol=1e-8)
        else:
            return np.allclose(np.asarray(hand_pose_frame), 0.0, atol=1e-8)
        