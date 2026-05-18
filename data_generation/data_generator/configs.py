"""Configuration classes for data generation."""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class CameraConfig:
    """Camera intrinsic parameters configuration."""

    width: int = 1280
    height: int = 720
    fx: float = 385.51226806640625 * 2
    fy: float = 385.51226806640625 * 1.5
    cx: float = 323.8814392089844 * 2
    cy: float = 238.46090698242188 * 1.5
    #depth_scale: float = 0.0010000000474974513  # meters per unit
    depth_scale: float = 0.0005000000474974513

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Return the camera intrinsic matrix (3x3)."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class DetectionConfig:
    """Object detection configuration."""

    model_config_path: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    model_weights_path: str = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    text_prompt: str = "plate"
    box_threshold: float = 0.3
    text_threshold: float = 0.3


@dataclass
class HandConfig:
    """Hand model configuration."""

    right_hand_mesh_path: str = "handpoint.glb"
    left_hand_mesh_path: str = "left_handpoint.glb"
    scale: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.01]))

    # Pre-transformation offsets for each hand
    right_hand_pre_move: np.ndarray = field(
        default_factory=lambda: np.array([-0.02215015, -0.02799035, -0.01030822])
    )
    left_hand_pre_move: np.ndarray = field(
        default_factory=lambda: np.array([0.02215015, -0.02799035, -0.01030822])
    )

@dataclass
class GenerationConfig:
    """Data generation configuration."""

    data_root: str = "data/input"
    output_root: str = "data/output"

    # Episode range to process
    start_episode: int = 0
    end_episode: int = 5

    # Number of samples to generate per method
    num_grab_samples: int = 0
    num_grab_and_move_samples: int = 0
    num_grab_and_move_twice_samples: int = 0
    num_grab_and_move_thrice_samples: int = 0
    
    # New flexible tasks sample counts
    num_grab_one_samples: int = 5
    num_grab_two_samples: int = 5
    num_grab_three_samples: int = 5

    # Generation parameters
    max_attempts_multiplier: int = 5  # max_attempts = num_samples * multiplier
    enable_lift: bool = False # Whether to simulate the lifing effect during movement
    video_data: bool = False # Whether to save each frame or just the pointing frame

    def get_enabled_tasks(self) -> List[Tuple[str, int]]:
        """
        Get list of enabled tasks (tasks with num_samples > 0).
        
        Returns:
            List of (task_name, num_samples) tuples
        """
        tasks = []
        
        if self.num_grab_samples > 0:
            tasks.append(("grab", self.num_grab_samples))
        if self.num_grab_and_move_samples > 0:
            tasks.append(("grab_and_move", self.num_grab_and_move_samples))
        if self.num_grab_and_move_twice_samples > 0:
            tasks.append(("grab_and_move_twice", self.num_grab_and_move_twice_samples))
        if self.num_grab_and_move_thrice_samples > 0:
            tasks.append(("grab_and_move_thrice", self.num_grab_and_move_thrice_samples))
        if self.num_grab_one_samples > 0:
            tasks.append(("grab_one", self.num_grab_one_samples))
        if self.num_grab_two_samples > 0:
            tasks.append(("grab_two", self.num_grab_two_samples))
        if self.num_grab_three_samples > 0:
            tasks.append(("grab_three", self.num_grab_three_samples))
            
        return tasks



@dataclass
class InstructionTemplates:
    """Instruction templates for different tasks."""

    grab: List[str] = field(default_factory=lambda: [
        "Instruction: pick this up.",
        "Instruction: grab this one.",
        "Instruction: take this.",
        "Instruction: lift this object.",
        "Instruction: pick up this item.",
    ])

    grab_and_move: List[str] = field(default_factory=lambda: [
        "Instruction: pick this up and put it there.",
        "Instruction: take this one and place it on that.",
        "Instruction: pick it up and put it over there.",
        "Instruction: carry this and release it there.",
        "Instruction: stack this block over there.",
        "Instruction: move this onto that one.",
    ])

    grab_and_move_twice: List[str] = field(default_factory=lambda: [
        "Instruction: pick up this and that, and put them there.",
        "Instruction: take these two and place them on that.",
        "Instruction: grab both of these and put them over there.",
        "Instruction: carry these two items and release them there.",
        "Instruction: move these two objects onto that plate.",
    ])

    grab_and_move_thrice: List[str] = field(default_factory=lambda: [
        "Instruction: pick up these three and put them there.",
        "Instruction: take these three items and place them on that.",
        "Instruction: grab all three and put them over there.",
        "Instruction: carry these three objects and release them there.",
        "Instruction: move these three objects onto that plate.",
    ])

    # New flexible tasks - select from all detected objects
    grab_one: List[str] = field(default_factory=lambda: [
        "Instruction: give me this.",
        "Instruction: pick ip this.",
        "Instruction: give me this one.",
    ])

    grab_two: List[str] = field(default_factory=lambda: [
        "Instruction: give me this and that.",
        "Instruction: pick up these.",
        "Instruction: give me these two.",
    ])

    grab_three: List[str] = field(default_factory=lambda: [
        "Instruction: give me these three.",
        "Instruction: give me this, that, and the other one.",
        "Instruction: pick up these three.",
    ])

    