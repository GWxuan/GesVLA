
from collections.abc import Sequence
import logging
import os
import re
from typing import Any, TypeAlias
import threading

import jax
import jax.numpy as jnp
import numpy as np
import torch
import cv2
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model_2vlm as _model
from openpi.models.pi0_ges_2vlm import Pi0Ges
from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi.shared import image_tools
from openpi.policies.video_process_multiframe import process_video_frames

BasePolicy: TypeAlias = _base_policy.BasePolicy
logger = logging.getLogger("openpi.ges_policy_2vlm_withref")

PALIGEMMA_EOS_TOKEN = 1


class GesPolicy2VLMWithRef(BasePolicy):

    def __init__(
        self,
        model: Pi0Ges,
        *,
        rng: at.KeyArrayLike | None = None,
        input_transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.model = model

        self._prefill = nnx_utils.module_jit(
            model.prefill,
            static_argnames=('temprature',),
        )
        self._act = nnx_utils.module_jit(model.act)

        # Use the 2VLM model's own VLM0 for reasoning (Phase 2).
        self._reason = nnx_utils.module_jit(
            model.reason,
            static_argnames=('temprature',),
        )

        # Data transform pipelines
        self._input_transform = _transforms.compose(input_transforms)
        self._output_transform = _transforms.compose(output_transforms)

        # RNG state
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        # Temperature for sampling
        self._temperature = 0.0

        # Tokenizer (decode VLM0-generated tokens)
        self._tokenizer = _tokenizer.FusePaligemmaTokenizer(
            max_len=model.max_token_len,
        )

        # Gesture cache
        self._gesture_cache = {
            "ges_images": None,
            "ges_image_mask": None,
            "hand_pose": None,
            "hand_pose_mask": None,
            "thought": None,
            "valid": False,
        }
        self._cache_lock = threading.Lock()

        # Reference image state
        self._reference_image: np.ndarray | None = None   # 224x224 uint8
        self._has_reference_image: bool = False
        self._generated_reasoning: str | None = None
        # "idle"      : just initialized / cleared
        # "reasoning" : gesture cached, waiting for reasoning
        # "action"    : reference image prepared, action-only inference
        self._phase: str = "idle"

        logger.info("GesPolicy2VLMWithRef initialized")

    # ------------------------------------------------------------------
    # Gesture video processing (same as ges_policy_2vlm.py)
    # ------------------------------------------------------------------
    def _prepare_gesture_data(self, obs: dict) -> dict:
        import tempfile

        logger.info("Starting gesture video processing")

        video_data = obs.get("video_data")
        thought = obs.get("thought")

        if video_data is None:
            logger.error("obs is missing 'video_data'")
            return {"status": "error", "message": "Missing video_data"}

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            tmp_file.write(video_data)
            tmp_path = tmp_file.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
        finally:
            os.unlink(tmp_path)
        logger.info("Read %d frames from video", len(frames))
        if len(frames) == 0:
            logger.error("Video contains zero frames")
            return {"status": "error", "message": "Video contains zero frames"}

        pause_frames, keypoint_vectors = process_video_frames(frames)

        if pause_frames is None or keypoint_vectors is None:
            logger.warning("Video processing failed; using defaults")
            pause_frames = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(3)]
            keypoint_vectors = [np.zeros(12, dtype=np.float32) for _ in range(3)]

        # Build gesture image dict
        ges_image_dict = {}
        ges_image_mask_dict = {}
        for i, frame in enumerate(pause_frames[:3]):
            key = f"ges_{i}_rgb"
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            ges_image_dict[key] = rgb_frame
            ges_image_mask_dict[key] = True
        for i in range(len(pause_frames), 3):
            key = f"ges_{i}_rgb"
            ges_image_dict[key] = np.zeros((224, 224, 3), dtype=np.uint8)
            ges_image_mask_dict[key] = False

        # Build hand keypoints
        num_frames = 3
        if len(keypoint_vectors) >= num_frames:
            hand_pose_array = np.array(keypoint_vectors[:num_frames], dtype=np.float32)
            hand_pose_mask = np.ones(num_frames, dtype=bool)
        else:
            hand_pose_array = np.zeros((num_frames, 12), dtype=np.float32)
            for i in range(min(len(keypoint_vectors), num_frames)):
                hand_pose_array[i] = keypoint_vectors[i]
            hand_pose_mask = np.array(
                [True] * len(keypoint_vectors) + [False] * (num_frames - len(keypoint_vectors)),
                dtype=bool,
            )

        if isinstance(thought, str):
            thought_list = [thought]
        else:
            thought_list = thought

        # Update cache
        with self._cache_lock:
            self._gesture_cache["ges_images"] = ges_image_dict
            self._gesture_cache["ges_image_mask"] = ges_image_mask_dict
            self._gesture_cache["hand_pose"] = torch.from_numpy(hand_pose_array)
            self._gesture_cache["hand_pose_mask"] = torch.from_numpy(hand_pose_mask)
            self._gesture_cache["thought"] = thought_list
            self._gesture_cache["valid"] = True

        # Reset reference image state
        self._reference_image = None
        self._has_reference_image = False
        self._generated_reasoning = None
        self._phase = "reasoning"

        logger.info(
            "Gesture cache updated: %d frames, thought=%s",
            len(pause_frames),
            thought_list,
        )

        # Optional debug dump for gesture frames
        debug_dir = os.environ.get("OPENPI_GESTURE_DEBUG_DIR")
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            for i, (key, img) in enumerate(ges_image_dict.items()):
                cv2.imwrite(
                    os.path.join(debug_dir, f"ges_frame_{i}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    if ges_image_mask_dict.get(key, False)
                    else img,
                )

        return {
            "status": "success",
            "message": "Gesture data cached; waiting for reasoning",
            "num_frames": len(pause_frames),
            "thought": thought_list,
        }

    # ------------------------------------------------------------------
    # Prepare observation data (2VLM action format)
    # ------------------------------------------------------------------
    def _prepare_observation(self, obs: dict, *, include_reference: bool = False) -> dict:
        """
        Prepare model input observations.

        Args:
            obs: Raw observation dictionary.
            include_reference: Whether to add cached reference_image to VLM1 inputs.
        """
        logger.debug("Observation keys: %s", list(obs.keys()))
        return_dict: dict[str, Any] = {}

        # 1) VLM1 action images
        image_dict: dict[str, Any] = {}
        image_mask_dict: dict[str, Any] = {}

        wrist_img = obs.get("observation/wrist_image")
        global_img = obs.get("observation/global_image")
        right_img = obs.get("observation/right_image")

        if wrist_img is not None and global_img is not None and right_img is not None:
            image_dict["global_image_1"] = global_img
            image_dict["right_image_1"] = right_img
            image_dict["wrist_image_1"] = wrist_img
            image_mask_dict["global_image_1"] = True
            image_mask_dict["right_image_1"] = True
            image_mask_dict["wrist_image_1"] = True
        else:
            logger.warning("Missing action images; using placeholders")
            for key in ["global_image_1", "right_image_1", "wrist_image_1"]:
                image_dict[key] = np.zeros((224, 224, 3), dtype=np.uint8)
                image_mask_dict[key] = False

        # Optional: add reference_image
        if include_reference and self._has_reference_image and self._reference_image is not None:
            image_dict["reference_image"] = self._reference_image   # 已经是 224×224 uint8
            image_mask_dict["reference_image"] = True

        return_dict["image"] = image_dict
        return_dict["image_mask"] = image_mask_dict

        # 2) VLM0 gesture data
        with self._cache_lock:
            if self._gesture_cache["valid"]:
                return_dict["ges_image"] = self._gesture_cache["ges_images"].copy()
                return_dict["ges_image_mask"] = self._gesture_cache["ges_image_mask"].copy()
                return_dict["hand_pose"] = self._gesture_cache["hand_pose"].clone()
                return_dict["hand_pose_mask"] = self._gesture_cache["hand_pose_mask"].clone()
                return_dict["thought"] = self._gesture_cache["thought"]
            else:
                ges_image_dict = {}
                ges_image_mask_dict = {}
                for i in range(3):
                    key = f"ges_{i}_rgb"
                    ges_image_dict[key] = np.zeros((224, 224, 3), dtype=np.uint8)
                    ges_image_mask_dict[key] = False
                return_dict["ges_image"] = ges_image_dict
                return_dict["ges_image_mask"] = ges_image_mask_dict
                return_dict["hand_pose"] = torch.zeros((3, 12), dtype=torch.float32)
                return_dict["hand_pose_mask"] = torch.zeros(3, dtype=torch.bool)
                return_dict["thought"] = ["Instruction: follow the gesture."]

        # 3) State
        state = obs.get("state", np.zeros(self.model.action_dim, dtype=np.float32))
        state = np.asarray(state, dtype=np.float32)
        if len(state) < self.model.action_dim:
            state = np.pad(state, (0, self.model.action_dim - len(state)))
        return_dict["state"] = torch.from_numpy(state[: self.model.action_dim])

        # 4) Action placeholders
        return_dict["actions"] = torch.zeros(
            (self.model.action_horizon, self.model.action_dim), dtype=torch.float32,
        )
        return_dict["action_is_pad"] = torch.ones(self.model.action_horizon, dtype=torch.bool)

        return return_dict

    # ------------------------------------------------------------------
    #  Token 解码
    # ------------------------------------------------------------------
    def _decode_tokens(self, tokens: np.ndarray) -> str:
        """Decode a token sequence into reasoning text."""
        token_array = np.asarray(tokens)
        if token_array.ndim > 1:
            token_array = token_array[0]

        # Truncate at EOS
        eos_positions = np.where(token_array == PALIGEMMA_EOS_TOKEN)[0]
        if len(eos_positions) > 0:
            valid_tokens = token_array[: eos_positions[0] + 1]
        else:
            valid_tokens = token_array
            logger.warning("EOS token not found; decoding full sequence")

        try:
            text = self._tokenizer.extract_thoughts(valid_tokens)
            if not text or text.strip() == "":
                text = f"Generated tokens: {valid_tokens[:20]}"
        except Exception as e:
            logger.error("Failed to decode tokens: %s", e)
            text = f"Decode error: {e}"

        return text

    @staticmethod
    def _clean_reasoning_text(text: str) -> str:
        """Remove <locXXX> tags from reasoning text."""
        return re.sub(r'<loc\d+>', '', text).strip()

    # ------------------------------------------------------------------
    # Generate reference_image
    # ------------------------------------------------------------------
    def _make_reference_image(self, raw_right_image: np.ndarray, reasoning_text: str) -> np.ndarray | None:
        """
        Use visualprompt to draw points on the right image and return a 224×224 uint8 image.

        Note: visualprompt keeps the input color space (no RGB/BGR conversion).
        We keep that behavior to match the training pipeline.

        Args:
            raw_right_image: Original right_image (any resolution, caller-defined color space).
            reasoning_text: Text containing (x,y) coordinate pairs.
        Returns:
            A 224×224 uint8 image or None.
        """
        from openpi.policies.reasoning_visualize import visualprompt

        ref_img = visualprompt(raw_right_image, reasoning_text)
        if ref_img is None:
            logger.warning("visualprompt returned None; failed to build reference_image")
            return None

        # Resize to 224×224 (same as ResizeImages)
        ref_resized = image_tools.resize_with_pad(
            np.asarray(ref_img, dtype=np.uint8), 224, 224,
        )
        return ref_resized

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------
    def clear_cache(self):
        """Clear all caches (gesture + reference_image)."""
        with self._cache_lock:
            self._gesture_cache = {
                "ges_images": None,
                "ges_image_mask": None,
                "hand_pose": None,
                "hand_pose_mask": None,
                "thought": None,
                "valid": False,
            }
        self._reference_image = None
        self._has_reference_image = False
        self._generated_reasoning = None
        self._phase = "idle"
        logger.info("All caches cleared")

    def is_cache_valid(self) -> bool:
        with self._cache_lock:
            return self._gesture_cache["valid"]

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------
    @override
    def infer(self, obs: dict) -> dict:
        """
        Main inference entry point.

        1) If obs includes video_data, process gesture video and cache it.
        2) In reasoning phase, use 2VLM VLM0 to generate reasoning text.
        3) In action phase, use 2VLM with reference_image to produce actions.
        """
        import time

        if "video_data" in obs:
            self._phase = "reasoning"
        else:
            self._phase = "action"

        # # ========== Phase 1: 手势视频上传 ==========
        # if "video_data" in obs:
        #     print("[WithRef] 检测到 video_data，进入手势处理流程")
        #     result = self._prepare_gesture_data(obs)
        #     return result

        # Phase 2: VLM0 reasoning → reasoning text
        if self._phase == "reasoning":
            logger.info("[WithRef] Detected video_data; processing gestures")
            result = self._prepare_gesture_data(obs)
            logger.info("[WithRef] Phase 2: VLM0 reasoning")
            time_start = time.time()

            inputs = self._prepare_observation(obs, include_reference=False)
            inputs = self._input_transform(inputs)
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            inputs = _model.FuseObservation.from_dict(inputs)

            prefill_rng, reason_rng, self._rng = jax.random.split(self._rng, 3)

            (
                processed_inputs,
                prefix_cache,
                first_suffix_token,
                eop_logit,
                prefix_mask,
                prefix_positions,
                has_boa,
                v1_len,
            ) = self._prefill(prefill_rng, inputs, temprature=self._temperature)

            time_prefill = time.time()
            logger.info("  Prefill time: %.4fs", time_prefill - time_start)

            suffix_tokens = self._reason(
                reason_rng,
                eop_logit,
                prefix_cache,
                prefix_mask,
                prefix_positions,
                v1_len,
                temprature=self._temperature,
            )

            time_reason = time.time()
            logger.info("  Reason time: %.4fs", time_reason - time_prefill)

            # Decode tokens → text
            generated_text = self._decode_tokens(suffix_tokens)
            generated_text = self._clean_reasoning_text(generated_text)
            logger.info("  Generated reasoning text: %s", generated_text)

            self._generated_reasoning = generated_text
            self._phase = "action"

            return {
                "status": "reasoning_completed",
                "generated_reasoning": generated_text,
                "has_reference_image": self._has_reference_image,
                "inference_time": time.time() - time_start,
                "gesture_processing": result,
            }

        # Phase 3: action inference (with reference_image)
        logger.info("[WithRef] Phase 3: action inference")
        time_start = time.time()

        # Prepare observation. Inject reference_image before transforms when available.
        inputs = self._prepare_observation(obs, include_reference=False)
        raw_right_image = inputs["image"].get("right_image_1")

        if (
            self._generated_reasoning
            and raw_right_image is not None
            and self._reference_image is None
        ):
            ref_img = self._make_reference_image(raw_right_image, self._generated_reasoning)
            if ref_img is not None:
                self._reference_image = ref_img
                self._has_reference_image = True
                logger.info("  reference_image generated successfully")
            else:
                logger.warning("  reference_image generation failed; continuing without it")

        if self._has_reference_image and self._reference_image is not None:
            inputs["image"]["reference_image"] = self._reference_image
            inputs["image_mask"]["reference_image"] = True

        inputs = self._input_transform(inputs)
        logger.debug("Input keys: %s", inputs.keys())

        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        inputs = _model.FuseObservation.from_dict(inputs)

        prefill_rng, act_rng, self._rng = jax.random.split(self._rng, 3)

        # Prefill
        (
            processed_inputs,
            prefix_cache,
            first_suffix_token,
            eop_logit,
            prefix_mask,
            prefix_positions,
            has_boa,
            v1_len,
        ) = self._prefill(prefill_rng, inputs, temprature=self._temperature)

        time_prefill = time.time()
        logger.info("  Prefill time: %.4fs", time_prefill - time_start)

        # Act (Action Expert denoising)
        actions = self._act(
            act_rng,
            processed_inputs,
            prefix_cache,
            prefix_mask,
            prefix_positions,
            v1_len,
        )

        time_act = time.time()
        logger.info("  Act time: %.4fs", time_act - time_prefill)

        # Build outputs
        outputs = {
            "state": inputs.state,
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Apply output transforms
        transformed = self._output_transform(outputs)
        logger.debug("Transformed outputs: %s", transformed)
        transformed["inference_time"] = time_act - time_start
        transformed["cache_valid"] = self.is_cache_valid()
        transformed["has_reference_image"] = self._has_reference_image
        transformed["generated_reasoning"] = self._generated_reasoning

        logger.info("  Inference completed in %.4fs", time_act - time_start)
        return transformed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def current_phase(self) -> str:
        return self._phase

    def get_generated_reasoning(self) -> str | None:
        return self._generated_reasoning
