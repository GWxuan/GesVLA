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
        # print("RepackTransform")
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        # print("InjectDefaultPrompt")
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
        norm = (x - stats.mean) / (stats.std + 1e-6)
        #print(f"Normalize: {norm[0]}")
        return norm

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
        # print("ResizeImage")
        resized = {}
        reference_params = None
        for k, v in data["image"].items():
            v = np.asarray(v)
            if v.dtype != np.uint8:
                v = v.astype(np.uint8)
            if v.shape[0] != self.height or v.shape[1] != self.width:
                #print(f"{k} resize")
                resized[k] = image_tools.resize_with_pad(v, self.height, self.width)
                if k == self.reference_frame:
                    #print("变换参数")
                    original_height, original_width = v.shape[:2]
                    reference_params = self._calculate_transform_params(
                        original_width, original_height, self.width, self.height
                    )
            else: resized[k] = v

        data["image"] = resized

        if "ges_images" in data:
            resized_ges = {}
            reference_params = None
            for k, v in data["ges_images"].items():
                v = np.asarray(v)
                if v.dtype != np.uint8:
                    v = v.astype(np.uint8)
                if v.shape[0] != self.height or v.shape[1] != self.width:
                    #print(f"{k} resize")
                    resized_ges[k] = image_tools.resize_with_pad(v, self.height, self.width)
                    if k == "ges_0_rgb":
                        #print("变换参数")
                        original_height, original_width = v.shape[:2]
                        reference_params = self._calculate_transform_params(
                            original_width, original_height, self.width, self.height
                        )
                else: resized_ges[k] = v
                data["ges_images"] = resized_ges

        
        if "thought" in data and reference_params is not None:
            # print(f"变换参数：{reference_params}")
            # print(f"变换前thought: {data['thought']}")
            data = self._transform_coordinates_in_thought(data, reference_params)
            #print(f"变换后thought: {data['thought']}")
        # import pdb; pdb.set_trace()
        if "hand_pose" in data and reference_params is not None:
            data = self._transform_hand_pose(data, reference_params)
        return data
    def _calculate_transform_params(self, orig_w: int, orig_h: int, target_w: int, target_h: int) -> dict:
        """计算图像变换参数"""
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
        """转换单个文本字符串中的坐标"""
        
        def replace_coordinate(match):
            x_orig = float(match.group(1))
            y_orig = float(match.group(2))
            
            # 转换坐标到新的图像空间
            x_new, y_new = self._transform_single_coordinate(
                x_orig, y_orig, transform_params
            )
            
            # 格式化为4位小数
            return f"({x_new:.4f}, {y_new:.4f})"
        
        # 使用正则表达式替换所有坐标
        return self.COORD_PATTERN.sub(replace_coordinate, text)
    
    def _transform_single_coordinate(self, x_orig: float, y_orig: float, params: dict) -> tuple[float, float]:
        """转换单个坐标"""
        
        # 原始像素坐标
        x_pixel = x_orig * params['original_width']
        y_pixel = y_orig * params['original_height']
        
        # 缩放后的坐标
        x_scaled = x_pixel * params['scale']
        y_scaled = y_pixel * params['scale']
        
        # 加上填充偏移
        x_padded = x_scaled + params['pad_left']
        y_padded = y_scaled + params['pad_top']
        
        # 归一化到新图像尺寸
        x_new = x_padded / params['target_width']
        y_new = y_padded / params['target_height']
        
        return x_new, y_new
    def _transform_hand_pose(self, data: DataDict, transform_params: dict) -> DataDict:
        """变换手势坐标"""
        hand_pose = data["hand_pose"]
        #print(f"hand_pose: {hand_pose.numpy()}")
        if isinstance(hand_pose, np.ndarray):
            hand_pose = hand_pose.copy()
        elif isinstance(hand_pose, torch.Tensor):
            hand_pose = hand_pose.clone()
        else:
            logging.warning("Unknown hand pose type: %s", type(hand_pose))
            return data
        
        
        if len(hand_pose.shape) == 2:  # [t, 12] 多个时间步
            for t in range(hand_pose.shape[0]):
                #print(t)
                if torch.allclose(hand_pose[t], torch.tensor(0.0), atol=1e-8):
                    continue
            
                # 处理4个关键点，每个关键点有3个坐标 (x, y, z)
                for i in range(4):  # 4个关键点
                    x_idx = i * 3
                    y_idx = i * 3 + 1
                    z_idx = i * 3 + 2
                    
                    # 获取原始归一化坐标
                    x_orig = hand_pose[t, x_idx]
                    y_orig = hand_pose[t, y_idx]
                    z = hand_pose[t, z_idx]
                    

                    # 变换 x, y 坐标
                    x_new, y_new = self._transform_single_coordinate(x_orig, y_orig, transform_params)
                    
                    # 更新变换后的坐标，z 坐标保持不变
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
        #print("GesTokenizePrompt")
        #print(f"275: datakey:{list(data.keys())}")
        #_ = data.pop("prompt", None)
        #print("ges tokenizer")
        thought = data.pop("thought", None)
        if thought is None:
            raise ValueError("Thought is required")
        (
            tokens,
            token_mask,
            ar_mask,
            text_loss_mask,
            diffusion_loss_mask,
        ) = self.tokenizer.tokenize(thought)
        
        

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
    """几何数据增强（随机裁剪和平移）"""
    apply_augmentation: bool = False  # 控制是否应用增强

    @timed_function("GeometricAugmentation")
    def __call__(self, data: DataDict) -> DataDict:
        if not self.apply_augmentation:
            #print("not self.apply_augmentation")
            return data
            
        import random
        
        # 随机选择一种增强方法
        aug_type = random.randint(0, 1)
        
        if aug_type == 0:
            # 随机裁剪
            crop_transform = RandomCropWithCoordSync(
                height=224,
                width=224,
                scale_range=(0.9, 1.0),
            )
            augmented_data = crop_transform(data)
            #print(f"✅ 应用随机裁剪，坐标已同步变换")
        else:
            # 随机平移
            translate_transform = RandomTranslateWithCoordSync(
                max_translate=20,
            )
            augmented_data = translate_transform(data)
            #print(f"✅ 应用随机平移，坐标已同步变换")
        
        return augmented_data


@dataclasses.dataclass(frozen=True)
class RandomCropWithCoordSync(DataTransformFn):
    """随机裁剪并同步变换坐标"""
    height: int
    width: int
    scale_range: tuple[float, float] = (0.9, 1.0)

    @timed_function("RandomCropWithCoordSync")
    def __call__(self, data: DataDict) -> DataDict:
            
        # 获取原图尺寸
        first_key = next(iter(data["image"].keys()))
        original_img = data["image"][first_key]
        orig_h, orig_w = original_img.shape[:2]
        
        # 计算裁剪尺寸
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        crop_h = int(orig_h * scale)
        crop_w = int(orig_w * scale)
        
        # 随机选择裁剪起始点
        start_y = np.random.randint(0, orig_h - crop_h + 1)
        start_x = np.random.randint(0, orig_w - crop_w + 1)

        reference_crop_h, reference_crop_w = 0, 0
        reference_start_y, reference_start_x = 0, 0
        
        # 裁剪并resize图像
        cropped_images = {}
        for key, img in data["image"].items():
            if "wrist" in key:
                cropped_images[key] = img
                continue
            if len(data["image"])==4:
                #print("随机变换",len(data["image"]))
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
                crop_h = int(orig_h * scale)
                crop_w = int(orig_w * scale)
                
                # 随机选择裁剪起始点
                start_y = np.random.randint(0, orig_h - crop_h + 1)
                start_x = np.random.randint(0, orig_w - crop_w + 1)
            if "reference" in key:
                reference_crop_h, reference_crop_w = crop_h, crop_w
                reference_start_y, reference_start_x = start_y, start_x
            # 裁剪
            cropped = img[start_y:start_y+crop_h, start_x:start_x+crop_w]
            # resize到目标尺寸
            resized = image_tools.resize_with_pad(cropped, self.height, self.width)
            cropped_images[key] = resized
            # self._save_debug_image(resized, f"crop_{int(time.time()*1000)}")

        data["image"] = cropped_images
        for key, img in data["ges_images"].items():
            cropped = img[reference_start_y:reference_start_y+reference_crop_h, reference_start_x:reference_start_x+reference_crop_w]

            resized = image_tools.resize_with_pad(cropped, self.height, self.width)
            data["ges_images"][key] = resized
            # self._save_debug_image(resized, f"crop_{int(time.time()*1000)}")
        # print(f"🔧 随机裁剪参数:")
        # print(f"   原图尺寸: {orig_w}x{orig_h}")
        # print(f"   裁剪尺寸: {crop_w}x{crop_h}")
        # print(f"   起始位置: ({start_x}, {start_y})")
        # print(f"   缩放比例: {scale:.3f}")

        
        #original_thought = data.get("thought", [])
        # original_hand_pose = data.get("hand_pose", None)
        
        # if original_thought:
        #     print(f"   变换前thought: {original_thought}")
        
        # if original_hand_pose is not None:
        #     print(f"   变换前hand_pose形状: {original_hand_pose.shape}")
        #     # 打印第一个时间步的前两个关键点
        #     if len(original_hand_pose.shape) >= 2:
        #         print(f"   变换前hand_pose[0, :6]: {original_hand_pose[0, :6]}")
        
        # 计算变换参数
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

        reference_transform_params = {
            'crop_start_x': reference_start_x,
            'crop_start_y': reference_start_y,
            'crop_width': reference_crop_w,
            'crop_height': reference_crop_h,
            'original_width': orig_w,
            'original_height': orig_h,
            'target_width': self.width,
            'target_height': self.height
        }

        # 变换图像
        
        # 变换thought中的坐标
        if "thought" in data:
            data = self._transform_thought_coordinates(data, reference_transform_params)
        
        # 变换hand_pose坐标
        if "hand_pose" in data:
            data = self._transform_hand_pose_for_crop(data, reference_transform_params)

        # transformed_thought = data.get("thought", [])
        # transformed_hand_pose = data.get("hand_pose", None)
        
        # if transformed_thought:
        #     print(f"   变换后thought: {transformed_thought}")
        
        # if transformed_hand_pose is not None and len(transformed_hand_pose.shape) >= 2:
        #     print(f"   变换后hand_pose[0, :6]: {transformed_hand_pose[0, :6]}")
        
        return data
    
    def _transform_thought_coordinates(self, data: DataDict, transform_params: dict) -> DataDict:
        """变换thought中的坐标"""
        transformed_thoughts = []
        
        for thought_text in data["thought"]:
            if isinstance(thought_text, str):           
                # 变换坐标
                transformed_text = self._transform_coordinates_in_text(thought_text, transform_params)
                transformed_thoughts.append(transformed_text)
            else:
                transformed_thoughts.append(thought_text)
        
        data["thought"] = transformed_thoughts
        return data
    
    def _transform_coordinates_in_text(self, text: str, transform_params: dict) -> str:
        """变换文本中的坐标"""
        coord_pattern = re.compile(r'\(([0-9.]+),\s*([0-9.]+)\)')
        
        def replace_coordinate(match):
            x_orig = float(match.group(1))
            y_orig = float(match.group(2))
            
            # 变换坐标
            x_new, y_new = self._transform_single_coordinate_for_crop(
                x_orig, y_orig, transform_params
            )
            
            return f"({x_new:.4f}, {y_new:.4f})"
        
        return coord_pattern.sub(replace_coordinate, text)
    
    def _transform_single_coordinate_for_crop(self, x_orig: float, y_orig: float, params: dict) -> tuple[float, float]:
        """为裁剪变换单个坐标"""
        # 原始像素坐标
        x_pixel = x_orig * params['original_width']
        y_pixel = y_orig * params['original_height']
        
        # 减去裁剪起始点
        x_cropped = x_pixel - params['crop_start_x']
        y_cropped = y_pixel - params['crop_start_y']
        
        # 归一化到裁剪区域
        x_normalized = x_cropped / params['crop_width']
        y_normalized = y_cropped / params['crop_height']
        
        # 归一化到目标尺寸（已经是[0,1]范围）
        return x_normalized, y_normalized
    
    def _transform_hand_pose_for_crop(self, data: DataDict, transform_params: dict) -> DataDict:
        """为裁剪变换手势坐标"""
        hand_pose = data["hand_pose"]
        
        if isinstance(hand_pose, np.ndarray):
            hand_pose = hand_pose.copy()
        elif isinstance(hand_pose, torch.Tensor):
            hand_pose = hand_pose.clone()
        else:
            return data
        
        # 手势坐标形状检查
        if len(hand_pose.shape) == 1:  # [12] 单个时间步
            hand_pose = hand_pose.reshape(1, -1)
        
        if len(hand_pose.shape) == 2:  # [t, 12] 多个时间步
            for t in range(hand_pose.shape[0]):
                # 检查是否为全零填充
                if self._is_zero_padding(hand_pose[t]):
                    continue
                
                # 处理4个关键点
                for i in range(4):
                    x_idx = i * 3
                    y_idx = i * 3 + 1
                    z_idx = i * 3 + 2
                    
                    x_orig = hand_pose[t, x_idx]
                    y_orig = hand_pose[t, y_idx]
                    
                    # 变换x, y坐标
                    x_new, y_new = self._transform_single_coordinate_for_crop(
                        x_orig, y_orig, transform_params
                    )
                    
                    hand_pose[t, x_idx] = x_new
                    hand_pose[t, y_idx] = y_new
        
        data["hand_pose"] = hand_pose
        return data
    
    def _is_zero_padding(self, hand_pose_frame: np.ndarray | torch.Tensor) -> bool:
        """检查手势坐标帧是否为全零填充"""
        if isinstance(hand_pose_frame, np.ndarray):
            return np.allclose(hand_pose_frame, 0.0, atol=1e-8)
        elif isinstance(hand_pose_frame, torch.Tensor):
            return torch.allclose(hand_pose_frame, torch.tensor(0.0), atol=1e-8)
        else:
            return np.allclose(np.asarray(hand_pose_frame), 0.0, atol=1e-8)
@dataclasses.dataclass(frozen=True)
class RandomTranslateWithCoordSync(DataTransformFn):
    """随机平移并同步变换坐标"""
    max_translate: int = 20  # 最大平移像素数

    @timed_function("RandomTranslateWithCoordSync")
    def __call__(self, data: DataDict) -> DataDict:
            
        # 随机生成平移量
        tx = np.random.randint(-self.max_translate, self.max_translate + 1)
        ty = np.random.randint(-self.max_translate, self.max_translate + 1)
        
        if tx == 0 and ty == 0:
            return data  # 没有平移

        # print(f"🔧 随机平移参数:")
        # print(f"   平移量: ({tx}, {ty}) 像素")
        
        # 平移图像
        reference_tx, reference_ty = 0, 0
        translated_images = {}
        for key, img in data["image"].items():
            if "wrist" in key:
                translated_images[key] = img
                continue
            if len(data["image"])==4:
                #print("随机变换",len(data["image"]))            
                tx = np.random.randint(-self.max_translate, self.max_translate + 1)
                ty = np.random.randint(-self.max_translate, self.max_translate + 1)
            if "reference" in key:
                reference_tx, reference_ty = tx, ty
            h, w = img.shape[:2]
            translated = self._translate_image(img, tx, ty)
            translated_images[key] = translated
            # self._save_debug_image(translated, f"translate_{int(time.time()*1000)}")
        
        data["image"] = translated_images

        for key, img in data["ges_images"].items():
            h, w = img.shape[:2]
            translated = self._translate_image(img, reference_tx, reference_ty)
            data["ges_images"][key] = translated
            # self._save_debug_image(translated, f"translate_{int(time.time()*1000)}")
        
        # 变换参数
        transform_params = {
            'translate_x': tx,
            'translate_y': ty,
            'image_width': w,
            'image_height': h
        }

        transform_ges_params = {
            'translate_x': reference_tx,
            'translate_y': reference_ty,
            'image_width': w,
            'image_height': h
        }

        # original_thought = data.get("thought", [])
        original_hand_pose = data.get("hand_pose", None)
        
        # if original_thought:
        #     print(f"   变换前thought: {original_thought}")
        
        # if original_hand_pose is not None:
        #     print(f"   变换前hand_pose形状: {original_hand_pose.shape}")
        #     # 打印第一个时间步的前两个关键点
        #     if len(original_hand_pose.shape) >= 2:
        #         print(f"   变换前hand_pose[0, :6]: {original_hand_pose[0, :6]}")
        
        # 变换thought中的坐标
        if "thought" in data:
            data = self._transform_thought_coordinates_for_translate(data, transform_params)
        
        # 变换hand_pose坐标
        if "hand_pose" in data:
            data = self._transform_hand_pose_for_translate(data, transform_ges_params)

        # transformed_thought = data.get("thought", [])
        # transformed_hand_pose = data.get("hand_pose", None)
        
        # if transformed_thought:
        #     print(f"   变换后thought: {transformed_thought}")
        
        # if transformed_hand_pose is not None and len(transformed_hand_pose.shape) >= 2:
        #     print(f"   变换后hand_pose[0, :6]: {transformed_hand_pose[0, :6]}")
        
        return data
    
    def _translate_image(self, img: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """平移图像 - 正确实现"""
        h, w = img.shape[:2]
        translated = np.zeros_like(img)
        
        # 计算平移后的有效区域
        if tx >= 0 and ty >= 0:
            # 向右下平移：原图左上部分移动到新图右下，新图左上补零
            src_x1, src_y1 = 0, 0
            src_x2, src_y2 = w - tx, h - ty
            dst_x1, dst_y1 = tx, ty
            dst_x2, dst_y2 = w, h
        elif tx >= 0 and ty < 0:
            # 向右上平移：原图左下部分移动到新图右上，新图左下补零
            src_x1, src_y1 = 0, -ty
            src_x2, src_y2 = w - tx, h
            dst_x1, dst_y1 = tx, 0
            dst_x2, dst_y2 = w, h + ty
        elif tx < 0 and ty >= 0:
            # 向左下平移：原图右上部分移动到新图左下，新图右上补零
            src_x1, src_y1 = -tx, 0
            src_x2, src_y2 = w, h - ty
            dst_x1, dst_y1 = 0, ty
            dst_x2, dst_y2 = w + tx, h
        else:  # tx < 0 and ty < 0
            # 向左上平移：原图右下部分移动到新图左上，新图右下补零
            src_x1, src_y1 = -tx, -ty
            src_x2, src_y2 = w, h
            dst_x1, dst_y1 = 0, 0
            dst_x2, dst_y2 = w + tx, h + ty
        
        # 确保坐标在有效范围内
        src_x1, src_y1 = max(0, src_x1), max(0, src_y1)
        src_x2, src_y2 = min(w, src_x2), min(h, src_y2)
        dst_x1, dst_y1 = max(0, dst_x1), max(0, dst_y1)
        dst_x2, dst_y2 = min(w, dst_x2), min(h, dst_y2)
        
        # 复制图像数据
        if src_x2 > src_x1 and src_y2 > src_y1:
            translated[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
        
        return translated
        
    def _transform_thought_coordinates_for_translate(self, data: DataDict, transform_params: dict) -> DataDict:
        """为平移变换thought中的坐标"""
        transformed_thoughts = []
        
        for thought_text in data["thought"]:
            if isinstance(thought_text, str):
                
                # 变换坐标
                transformed_text = self._transform_coordinates_in_text_for_translate(thought_text, transform_params)
                transformed_thoughts.append(transformed_text)
            else:
                transformed_thoughts.append(thought_text)
        
        data["thought"] = transformed_thoughts
        return data
    
    def _transform_coordinates_in_text_for_translate(self, text: str, transform_params: dict) -> str:
        """为平移变换文本中的坐标"""
        coord_pattern = re.compile(r'\(([0-9.]+),\s*([0-9.]+)\)')
        
        def replace_coordinate(match):
            x_orig = float(match.group(1))
            y_orig = float(match.group(2))
            
            # 变换坐标
            x_new, y_new = self._transform_single_coordinate_for_translate(
                x_orig, y_orig, transform_params
            )
            
            return f"({x_new:.4f}, {y_new:.4f})"
        
        return coord_pattern.sub(replace_coordinate, text)
    
    def _transform_single_coordinate_for_translate(self, x_orig: float, y_orig: float, params: dict) -> tuple[float, float]:
        """为平移变换单个坐标 - 正确实现"""
        # 原始像素坐标
        x_pixel = x_orig * params['image_width']
        y_pixel = y_orig * params['image_height']
        
        # 图像平移方向与坐标移动方向相同
        # 如果图像向左平移，坐标也向左移动（x减少）
        # 如果图像向下平移，坐标也向下移动（y增加）
        x_translated = x_pixel + params['translate_x']
        y_translated = y_pixel + params['translate_y']
        
        # 归一化回[0,1]范围
        x_normalized = x_translated / params['image_width']
        y_normalized = y_translated / params['image_height']
        
        return x_normalized, y_normalized
    
    def _transform_hand_pose_for_translate(self, data: DataDict, transform_params: dict) -> DataDict:
        """为平移变换手势坐标"""
        hand_pose = data["hand_pose"]
        
        if isinstance(hand_pose, np.ndarray):
            hand_pose = hand_pose.copy()
        elif isinstance(hand_pose, torch.Tensor):
            hand_pose = hand_pose.clone()
        else:
            return data
        
        # 手势坐标形状检查
        if len(hand_pose.shape) == 1:
            hand_pose = hand_pose.reshape(1, -1)
        
        if len(hand_pose.shape) == 2:
            for t in range(hand_pose.shape[0]):
                # 检查是否为全零填充
                if self._is_zero_padding(hand_pose[t]):
                    continue
                
                # 处理4个关键点
                for i in range(4):
                    x_idx = i * 3
                    y_idx = i * 3 + 1
                    z_idx = i * 3 + 2
                    
                    x_orig = hand_pose[t, x_idx]
                    y_orig = hand_pose[t, y_idx]
                    
                    # 变换x, y坐标
                    x_new, y_new = self._transform_single_coordinate_for_translate(
                        x_orig, y_orig, transform_params
                    )
                    
                    hand_pose[t, x_idx] = x_new
                    hand_pose[t, y_idx] = y_new
        
        data["hand_pose"] = hand_pose
        return data
    
    def _is_zero_padding(self, hand_pose_frame: np.ndarray | torch.Tensor) -> bool:
        """检查手势坐标帧是否为全零填充"""
        if isinstance(hand_pose_frame, np.ndarray):
            return np.allclose(hand_pose_frame, 0.0, atol=1e-8)
        elif isinstance(hand_pose_frame, torch.Tensor):
            return torch.allclose(hand_pose_frame, torch.tensor(0.0), atol=1e-8)
        else:
            return np.allclose(np.asarray(hand_pose_frame), 0.0, atol=1e-8)

