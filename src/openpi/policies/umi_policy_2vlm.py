import dataclasses

import einops
import numpy as np

from openpi import transforms_2vlm as transforms
from openpi.models import model_2vlm as _model


def make_umi_example() -> dict:
    """Creates a random input example for the umi policy."""
    return {
        "state": np.random.rand(48),
        "image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UMIInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0_GES

    def __call__(self, data: dict) -> dict:
        #print("umi_policy")
        mask_padding = self.model_type == _model.ModelType.PI0_GES  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        history_length = 1
        while True:
            if f"global_image_{history_length + 1}" not in data:
                break
            history_length += 1
        
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        image_dict, image_mask_dict, image_ges_dict, image_ges_mask_dict = {}, {},{}, {}
        # import pdb;pdb.set_trace()
        for i in range(history_length):
           
            image_dict[f"{i}_global_rgb"] = _parse_image(data['image'][f"global_image_{i+1}"])
            image_dict[f"{i}_right_rgb"] = _parse_image(data['image'][f"right_image_{i+1}"])
            image_dict[f"{i}_wrist_rgb"] = _parse_image(data['image'][f"wrist_image_{i+1}"])


            image_mask_dict[f"{i}_rgb"] = np.True_

        image_dict['reference_image'] = _parse_image(data['image']['reference_image'])
        image_mask_dict['reference_image'] = np.True_

        image_ges_dict["ges_0_rgb"] = _parse_image(data['ges_image']["ges_0_rgb"])
        image_ges_dict["ges_1_rgb"] = _parse_image(data['ges_image']["ges_1_rgb"])
        image_ges_dict["ges_2_rgb"] = _parse_image(data['ges_image']["ges_2_rgb"])
        image_ges_mask_dict = data['ges_image_mask']
        # import pdb;pdb.set_trace()
        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict,
            "ges_images": image_ges_dict,
            "ges_image_masks": image_ges_mask_dict,
            "hand_pose": data['hand_pose'],
            "hand_pose_mask": data['hand_pose_mask'],
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if 'thought' in data.keys():
            inputs['thought'] = data['thought']
            inputs['act_with_outdated_thought'] = data.get('act_with_outdated_thought', False)
            inputs['think_with_outdated_thought'] = data.get('think_with_outdated_thought', False)
        if 'origin_episode_idx' in data.keys():
            inputs['origin_episode_idx'] = data['origin_episode_idx']
            inputs['idx'] = data['index']
        # import pdb;pdb.set_trace()
        return inputs


@dataclasses.dataclass(frozen=True)
class UMIOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 10 dims.
        data.update({"actions": np.asarray(data["actions"][:, :7])})
        return data

