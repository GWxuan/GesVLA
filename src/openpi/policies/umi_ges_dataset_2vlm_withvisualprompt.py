from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import copy
import io
import json
import logging
import os
import re

import numpy as np
import pyarrow.parquet as pq
import scipy.interpolate as si
import scipy.spatial.transform as st
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

from openpi.policies.pose_util import pose_to_mat, mat_to_pose10d, mat_to_pose
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.training.gesconfig_2vlm import UMIDataConfig
from openpi.timer_utils import Timer, timed_function
from openpi import transforms_2vlm as transforms
import random

from PIL import Image


content_grab_and_move = [
    "Instruction: pick this up and put it there.",
    "Instruction: take this one and place it on that.",
    "Instruction: pick it up and put it over there.",
    "Instruction: carry this and release it there.",
    "Instruction: stack this block over there.",
    "Instruction: move this onto that one.",
]

def load_entire_dataset(parquet_path):
    """Read a parquet file in batches and aggregate rows into a list."""
    parquet_file = pq.ParquetFile(parquet_path)
    all_data = []
    total_rows = parquet_file.metadata.num_rows
    batch_size = 1000
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_data = batch.to_pylist()
        all_data.extend(batch_data)
    logger.info("Dataset loaded: %d rows", len(all_data))
    return all_data

from typing import Any, Dict, List
import os
import json
import numpy as np
import torch
import hashlib
import cv2

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Padding helper from transforms.
from openpi import transforms_2vlm as transforms


class UMIGesDataset_2vlm(LeRobotDataset):
    """Each origin_episode_idx has a single row in the gesture parquet file:

    The row contains image (list[bytes]) / point_feature (list[t,12]).
    Therefore, __getitem__ does not need intra-episode indexing and
    returns the same gesture info for each action frame.
    """

    def __init__(self, data_config, action_horizon: int):
        self.data_config = data_config
        self.action_horizon = action_horizon

        repo_id = str(data_config.repo_id)
        root = str(data_config.root)
        logger.info("Loading LeRobotDataset: repo_id=%s, root=%s", repo_id, root)
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root)

        # ===== 1) Load gesture parquet =====
        self.ges_parquet_path = getattr(data_config, "ges_parquet_path", None)
        if self.ges_parquet_path is None:
            raise ValueError("ges_parquet_path is required in data_config")
        self.ges_rows = load_entire_dataset(self.ges_parquet_path)

        # ===== 2) Build origin_episode_idx -> row mapping (one row per episode) =====
        self.ges_by_origin = self._build_ges_map_single(self.ges_rows)

        # ===== Extra: reference image directory =====
        self.reference_image_dir = getattr(data_config, "reference_image_dir", None)
        if self.reference_image_dir is None:
            raise ValueError("reference_image_dir is required in data_config")

        # ===== 3) Episode boundaries (replace with the full UMI implementation if needed) =====
        self.episode_boundaries = self._build_episode_boundaries()
        self.indices = self._build_indices_all_frames()

    # -------------------------
    # Gesture parquet mapping: one row per origin_episode_idx
    # -------------------------
    def _build_ges_map_single(self, rows):
        """Build a mapping from origin_episode_idx to a single gesture row."""
        mp = {}
        dup = 0
        for r in rows:
            oid = r.get("origin_episode_idx", None)
            if oid is None:
                oid = r.get("episode_index", None)
            if oid is None:
                continue
            oid = int(oid)
            if oid in mp:
                dup += 1
                # If duplicates exist, keep the first entry by default.
                continue
            mp[oid] = r
        logger.info("GES parquet: %d origin episodes (%d duplicates skipped)", len(mp), dup)
        return mp

    def _placeholder_image(self):
        """Return a 224x224x3 zero image as a placeholder."""
        return np.zeros((224, 224, 3), dtype=np.uint8)

    def _load_reference_image(self, origin_episode_idx: int):
        """
        Load a reference image and return an HWC uint8 array.
        If the image is missing, return a placeholder.
        """
        reference_image_path = os.path.join(
            self.reference_image_dir, f"{origin_episode_idx}.png"
        )

        if not os.path.exists(reference_image_path):
            # Return a placeholder when the reference image is missing.
            return self._placeholder_image()

        try:
            ref_img = Image.open(reference_image_path).convert("RGB")
            ref_arr = np.array(ref_img, dtype=np.uint8)
            return ref_arr
        except Exception as e:
            logger.warning("Load reference image failed: %s, err=%s", reference_image_path, e)
            return self._placeholder_image()

    # -------------------------
    # Episode boundaries (minimal implementation; replace with UMI version for accuracy)
    # -------------------------
    def _build_episode_boundaries(self):
        """Compute per-episode start/end indices and origin mapping."""
        hf = self.dataset.hf_dataset
        episode_indices = hf["episode_index"]
        origin_episode_indices = hf["origin_episode_idx"]
        frame_indices = hf["frame_index"]

        episode_data = {}
        for gi in range(len(episode_indices)):
            ep = int(episode_indices[gi])
            oep = int(origin_episode_indices[gi])
            fi = int(frame_indices[gi])
            episode_data.setdefault(ep, {"origin": oep, "gis": [], "fis": []})
            episode_data[ep]["gis"].append(gi)
            episode_data[ep]["fis"].append(fi)

        boundaries = {}
        for ep, d in episode_data.items():
            gis = d["gis"]
            fis = d["fis"]
            boundaries[ep] = {
                "origin_episode_idx": d["origin"],
                "start_global_idx": min(gis),
                "end_global_idx": max(gis),
                "start_frame_idx": min(fis),
                "end_frame_idx": max(fis),
            }
        return boundaries

    def _get_episode_info(self, global_idx):
        """Return (start_idx, end_idx, episode_index) for a global index."""
        for ep, b in self.episode_boundaries.items():
            if b["start_global_idx"] <= global_idx <= b["end_global_idx"]:
                return b["start_global_idx"], b["end_global_idx"], ep
        last = max(self.episode_boundaries.keys())
        b = self.episode_boundaries[last]
        return b["start_global_idx"], b["end_global_idx"], last

    def _build_indices_all_frames(self):
        """Build the index list that iterates over all frames in all episodes."""
        out = []
        for _, b in self.episode_boundaries.items():
            s, e = b["start_global_idx"], b["end_global_idx"]
            if s <= e:
                out.extend(list(range(s, e)))  # Keep the original [s, e) behavior.
        logger.info("UMI total samples: %d", len(out))
        return out

    # -------------------------
    # decode helpers
    # -------------------------
    def _decode_png_bytes(self, image_bytes):
        """Decode PNG bytes into an HWC uint8 array."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(img)

    def _decode_image(self, image_data):
        """Decode image payloads that may wrap PNG bytes in a dict."""
        if isinstance(image_data, dict) and "bytes" in image_data:
            return self._decode_png_bytes(image_data["bytes"])
        return image_data

    def _ensure_hwc_uint8(self, x):
        """Convert tensors/arrays to HWC uint8 with 3 channels."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)

        # CHW -> HWC
        if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
            x = np.transpose(x, (1, 2, 0))

        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)

        if x.dtype != np.uint8:
            mx, mn = float(x.max()), float(x.min())
            if mx <= 1.0 and mn >= 0.0:
                x = (x * 255.0).clip(0, 255).astype(np.uint8)
            elif mx <= 1.0 and mn >= -1.0:
                x = ((x * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            else:
                x = x.clip(0, 255).astype(np.uint8)
        return x

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Return a training sample with images, gestures, and action targets."""
        original_idx = int(self.indices[idx])
        item = self.dataset[original_idx]

        start_idx, end_idx, episode_index = self._get_episode_info(original_idx)

        # Key: origin_episode_idx from the action dataset.
        origin_episode_idx = int(self.dataset.hf_dataset["origin_episode_idx"][original_idx])

        return_dict = {}

        # ---- No text training: provide a placeholder thought ----
        instruction = random.choice(content_grab_and_move)
        return_dict["thought"] = [instruction]

        # =========================================================
        # (1) VLM1 images: three views + reference image.
        # =========================================================
        img_global = self._ensure_hwc_uint8(self._decode_image(item["global_image"]))
        img_right  = self._ensure_hwc_uint8(self._decode_image(item["right_image"]))
        img_wrist  = self._ensure_hwc_uint8(self._decode_image(item["wrist_image"]))

        # Load the reference image.
        ref_img = self._load_reference_image(origin_episode_idx)

        return_dict["image"] = {
            "global_image_1": img_global,
            "wrist_image_1": img_wrist,
            "right_image_1": img_right,
            "reference_image": ref_img,   # added
        }
        return_dict["image_mask"] = {
            "global_image_1": True,
            "wrist_image_1": True,
            "right_image_1": True,
            "reference_image": True,      # added
        }

        # =========================================================
        # (2) Gesture images: one parquet row per origin_episode_idx.
        # =========================================================
        t_img = 3
        ges_image_dict = {}
        ges_image_mask = {}

        ges_row = self.ges_by_origin.get(origin_episode_idx, None)
        if ges_row is not None:
            imgs = ges_row.get("image", None)
            if isinstance(imgs, list) and len(imgs) > 0:
                actual = min(len(imgs), t_img)
                for i in range(actual):
                    arr = self._decode_png_bytes(imgs[i])
                    ges_image_dict[f"ges_{i}_rgb"] = arr
                    ges_image_mask[f"ges_{i}_rgb"] = True
                for i in range(actual, t_img):
                    ges_image_dict[f"ges_{i}_rgb"] = self._placeholder_image()
                    ges_image_mask[f"ges_{i}_rgb"] = False
            else:
                for i in range(t_img):
                    ges_image_dict[f"ges_{i}_rgb"] = self._placeholder_image()
                    ges_image_mask[f"ges_{i}_rgb"] = False
        else:
            # No gesture data for this origin_episode: placeholder + False mask.
            for i in range(t_img):
                ges_image_dict[f"ges_{i}_rgb"] = self._placeholder_image()
                ges_image_mask[f"ges_{i}_rgb"] = False

        return_dict["ges_image"] = ges_image_dict
        return_dict["ges_image_mask"] = ges_image_mask

        # =========================================================
        # (3) Hand pose: one parquet row per origin_episode_idx.
        # =========================================================
        t_pose = 3
        hp = np.zeros((t_pose, 12), dtype=np.float32)
        hp_mask = np.zeros((t_pose,), dtype=np.bool_)

        if ges_row is not None:
            pf = ges_row.get("point_feature", None)
            if pf is not None:
                pf = pf.tolist() if hasattr(pf, "tolist") else pf
                pf = list(pf) if isinstance(pf, (list, tuple)) else []
                valid = min(len(pf), t_pose)
                if valid > 0:
                    hp[:valid] = np.asarray(pf[:valid], dtype=np.float32)
                    hp_mask[:valid] = True

        return_dict["hand_pose"] = torch.from_numpy(hp)
        return_dict["hand_pose_mask"] = torch.from_numpy(hp_mask)

        # =========================================================
        # (4) state/actions/action_is_pad/diffusion_loss_mask
        # =========================================================
        return_dict["state"] = torch.zeros(32, dtype=torch.float32)

        action_start = original_idx
        action_end = min(end_idx, action_start + self.action_horizon)

        actions_list = []
        for act_idx in range(action_start, action_end):
            act_item = self.dataset[int(act_idx)]
            a = act_item["actions"]
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            actions_list.append(np.asarray(a, dtype=np.float32))

        if len(actions_list) > 0:
            actions = np.stack(actions_list, axis=0)
        else:
            actions = np.zeros((0, 7), dtype=np.float32)

        action_is_pad = [False] * len(actions) + [True] * (self.action_horizon - len(actions))
        return_dict["action_is_pad"] = torch.tensor(action_is_pad, dtype=torch.bool)

        if len(actions) < self.action_horizon:
            if len(actions) > 0:
                pad = np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)
                actions = np.concatenate([actions, pad], axis=0)
            else:
                actions = np.zeros((self.action_horizon, 7), dtype=np.float32)
        actions = actions[: self.action_horizon]

        actions_t = torch.from_numpy(actions.astype(np.float32))
        actions_t = transforms.pad_to_dim(actions_t, 32)
        return_dict["actions"] = actions_t

        return_dict["diffusion_loss_mask"] = torch.tensor([not all(action_is_pad)], dtype=torch.bool).squeeze(0)

        # metadata
        return_dict["timestamp"] = item.get("timestamp", 0.0)
        return_dict["frame_index"] = int(original_idx)
        return_dict["episode_index"] = int(episode_index)
        return_dict["origin_episode_idx"] = int(origin_episode_idx)
        return_dict["index"] = int(original_idx - start_idx)

        return return_dict