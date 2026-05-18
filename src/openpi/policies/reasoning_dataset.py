import copy
import io
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from openpi import transforms
from openpi.timer_utils import Timer, timed_function

logger = logging.getLogger("openpi")


def load_entire_dataset(parquet_path):
    """Load entire parquet file in batches."""
    parquet_file = pq.ParquetFile(parquet_path)
    all_data = []
    batch_size = 1000
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        all_data.extend(batch.to_pylist())
    logger.info("Loaded %d records from parquet", len(all_data))
    return all_data


class ReasoningDataset(Dataset):
    """Parquet-based dataset for gesture reasoning pre-training."""

    def __init__(self, data_config, action_horizon: int, split: str = "train"):
        self.data_config = data_config
        self.action_horizon = action_horizon
        self.split = split

        parquet_filename = getattr(data_config, "parquet_filename", "data.parquet")
        parquet_path = os.path.join(data_config.repo_id, parquet_filename)
        self.hf_dataset = load_entire_dataset(parquet_path)

        with open(data_config.reasoning_json_path, "r") as f:
            self.reasoning = json.load(f)

        self.getitem_type = getattr(data_config, "getitem_type", "necessary")
        self._create_split()

    def _create_split(self):
        total_size = len(self.hf_dataset)
        val_size = int(total_size * self.data_config.val_ratio)
        seed = getattr(self.data_config, "seed", 42)
        rng = np.random.RandomState(seed)
        indices = np.arange(total_size)

        if self.split == "train":
            train_indices = indices[val_size:]
            rng.shuffle(train_indices)
            self.indices = train_indices
        elif self.split == "test":
            self.indices = np.arange(total_size)
            logger.info("ReasoningDataset test: using ALL %d samples", len(self.indices))
        else:  # val
            val_indices = indices[:val_size]
            rng.shuffle(val_indices)
            self.indices = val_indices

        logger.info("ReasoningDataset %s: %d samples", self.split, len(self.indices))

    def __len__(self):
        return len(self.indices)

    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.split = "val"
        val_set._create_split()
        return val_set

    def _create_placeholder_image(self):
        return np.zeros((224, 224, 3), dtype=np.uint8)

    @timed_function("ReasoningDataset.__getitem__")
    def __getitem__(self, idx):
        with Timer("  -> current_item_load"):
            actual_idx = self.indices[idx]
            current = self.hf_dataset[actual_idx]

        with Timer("  -> reasoning_processing"):
            episode_id = str(current.get("episode_index", 0))
            return_dict = {}
            ep_data = self.reasoning.get(str(episode_id), {})
            segments = ep_data.get("segments", [])
            if segments:
                rd = segments[0]
                return_dict["thought"] = [rd["content"], rd["updated_content"]]
            else:
                return_dict["thought"] = ["", ""]

        with Timer("  -> image_processing"):
            imgs = current["image"]
            image_dict = {}
            image_mask_dict = {}
            actual_frame_count = len(imgs)

            for i, img_bytes in enumerate(imgs):
                img = Image.open(io.BytesIO(img_bytes))
                image_dict[f"{i}_rgb"] = np.array(img)
                image_mask_dict[f"{i}_rgb"] = True

            for i in range(actual_frame_count, 3):
                image_dict[f"{i}_rgb"] = self._create_placeholder_image()
                image_mask_dict[f"{i}_rgb"] = False

            return_dict["image"] = image_dict
            return_dict["image_mask"] = image_mask_dict

        with Timer("  -> action_processing"):
            freezing_action = [0., 0., 0., 1., 0., 0., 0., 1., 0., 0.08623]
            frz = torch.tensor(freezing_action, dtype=torch.float32)
            return_dict["actions"] = transforms.pad_to_dim(frz.unsqueeze(0).repeat(self.action_horizon, 1), 32)
            return_dict["action_is_pad"] = torch.tensor([True] * 32, dtype=torch.bool)

            if self.getitem_type == "necessary":
                rest_state = torch.tensor(
                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                     1., 0., 1., 0., 0., 0., 1., 0., 1., 0.,
                     0., 0., 1., 0., 0.08623, 0.08623, 0.08623],
                    dtype=torch.float32
                )
            state = transforms.pad_to_dim(rest_state, 32)
            return_dict["state"] = state

            if "point_feature" in current:
                point_features = current["point_feature"]
                if isinstance(point_features, list) and len(point_features) >= 3:
                    hand_pose_array = np.array(point_features[:3], dtype=np.float32)
                    return_dict["hand_pose"] = torch.from_numpy(hand_pose_array)
                    return_dict["hand_pose_mask"] = torch.ones(3, dtype=torch.bool)
                else:
                    hand_pose_array = np.zeros((3, 12), dtype=np.float32)
                    if len(point_features) > 0:
                        actual_len = min(len(point_features), 3)
                        hand_pose_array[:actual_len] = np.array(point_features[:actual_len], dtype=np.float32)
                    return_dict["hand_pose"] = torch.from_numpy(hand_pose_array)
                    return_dict["hand_pose_mask"] = torch.tensor(
                        [True] * min(len(point_features), 3) + [False] * max(0, 3 - len(point_features)),
                        dtype=torch.bool
                    )
            else:
                hand_pose_array = np.zeros((3, 12), dtype=np.float32)
                return_dict["hand_pose"] = torch.from_numpy(hand_pose_array)
                return_dict["hand_pose_mask"] = torch.tensor([False] * 3, dtype=torch.bool)

            for k in ["timestamp", "frame_index", "episode_index", "index", "task_index", "origin_episode_idx"]:
                if k in current:
                    return_dict[k] = current[k]

        return return_dict
