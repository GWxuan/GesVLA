"""Task-specific data generators."""

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


from data_generator.configs import CameraConfig, GenerationConfig, InstructionTemplates
from data_generator.phases import PhaseGenerator
from data_generator.renderer import HandMeshRenderer
from data_generator.utils import CoordinateUtils, HandColorGenerator, HandTransformer


class DataGeneratorBase(ABC):
    """Abstract base class for data generators."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer
    ):
        """
        Initialize the data generator.

        Args:
            camera_config: Camera configuration
            generation_config: Generation configuration
            instruction_templates: Instruction templates
            renderer: Hand mesh renderer
        """
        self.camera = camera_config
        self.gen_config = generation_config
        self.instructions = instruction_templates
        self.renderer = renderer
        self.coord_utils = CoordinateUtils(camera_config)
        self.transformer = HandTransformer()

    @abstractmethod
    def generate(self, *args, **kwargs) -> bool:
        """Generate data. Must be implemented by subclasses."""
        raise NotImplementedError

    def get_jittered_coords(
        self,
        box: np.ndarray,
        depth_image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Get 3D coordinates with random jitter.

        Args:
            box: Detection box (x_center, y_center, width, height)
            depth_image: Depth image

        Returns:
            Tuple of (3D_point, jittered_2d, real_2d)
        """
        x1, y1, x2, y2 = box

        x_jitter = x1 + random.uniform(-x2 / 4, x2 / 4)
        y_jitter = y1 + random.uniform(-y2 / 4, y2 / 4)

        x = int(x_jitter * self.camera.width)
        y = int(y_jitter * self.camera.height)
        x_real = int(x1 * self.camera.width)
        y_real = int(y1 * self.camera.height)

        x = np.clip(x, 0, self.camera.width - 1)
        y = np.clip(y, 0, self.camera.height - 1)
        x_real = np.clip(x_real, 0, self.camera.width - 1)
        y_real = np.clip(y_real, 0, self.camera.height - 1)

        center_3d = self.coord_utils.pixel_to_3d(x, y, depth_image)
        return center_3d, (x, y), (x_real, y_real)

    def normalize_2d_coords(self, coords_2d: Tuple[int, int]) -> Tuple[float, float]:
        """
        Normalize 2D coordinates to [0, 1] range.

        Args:
            coords_2d: Pixel coordinates (x, y)

        Returns:
            Normalized coordinates (x_norm, y_norm)
        """
        return (
            round(coords_2d[0] / self.camera.width, 4),
            round(coords_2d[1] / self.camera.height, 4)
        )

    def collect_object_coords(self, boxes: List[Tuple[np.ndarray, str]]) -> List[str]:
        """
        Collect normalized coordinates of all objects.

        Args:
            boxes: List of (box, phrase) tuples

        Returns:
            List of coordinate strings
        """
        coords = []
        for box, _ in boxes:
            center_x = int(box[0] * self.camera.width)
            center_y = int(box[1] * self.camera.height)
            center_x = np.clip(center_x, 0, self.camera.width - 1)
            center_y = np.clip(center_y, 0, self.camera.height - 1)
            center_x_norm = round(center_x / self.camera.width, 4)
            center_y_norm = round(center_y / self.camera.height, 4)
            coords.append(f"({center_x_norm}, {center_y_norm})")
        return coords

    def save_annotation(
        self,
        save_dir: Path,
        annotation: dict,
        annotation_with_coords: dict
    ) -> None:
        """
        Save annotation files.

        Args:
            save_dir: Directory to save to
            annotation: Basic annotation
            annotation_with_coords: Annotation with coordinates
        """
        detect_dir = save_dir / "detect"
        detect_dir.mkdir(parents=True, exist_ok=True)

        with open(detect_dir / "thought.json", "w") as f:
            json.dump(annotation, f, indent=4)

        with open(detect_dir / "thought_with_coordinate.json", "w") as f:
            json.dump(annotation_with_coords, f, indent=4)


class GrabGenerator(DataGeneratorBase):
    """Generator for grab task."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer,
        phase_generator: PhaseGenerator
    ):
        super().__init__(camera_config, generation_config, instruction_templates, renderer)
        self.phase_gen = phase_generator

    def generate(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        block_fruit_boxes: List[Tuple[np.ndarray, str]],
        plate_boxes: List[Tuple[np.ndarray, str]],
        save_dir_base: Path,
        episode_idx: int,
        idx: int
    ) -> bool:
        """Generate grab data."""
        try:
            method_name = "grab"
            save_dir = save_dir_base / method_name / f"{idx:02d}"
            (save_dir / "RGB").mkdir(parents=True, exist_ok=True)

            # Random select a block
            box_A, phrase_A = random.choice(block_fruit_boxes)
            objA, objA_2d, objA_2d_real = self.get_jittered_coords(box_A, depth_image)

            # Normalize coordinates
            objA_2d_norm = self.normalize_2d_coords(objA_2d)
            objA_2d_real_norm = self.normalize_2d_coords(objA_2d_real)

            # Collect all object coordinates
            all_block_coords = self.collect_object_coords(block_fruit_boxes)
            all_plate_coords = self.collect_object_coords(plate_boxes)

            hand_color = HandColorGenerator.get_random_color()
            step = 0

            # Phase 1: Approach object
            direction, hand_tip_pos, step, _ = self.phase_gen.generate_phase1(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objA, save_dir, step, episode_idx, idx, method_name, hand_color
            )

            if direction is None:
                return False

            # Create annotations
            instruction = random.choice(self.instructions.grab)
            annotation = {
                "0": {
                    "content": instruction + "\n",
                    "updated_content": f"I need to pick up the object located at {objA_2d_norm}.\n",
                }
            }

            block_coords_str = ", ".join(all_block_coords)
            plate_coords_str = ", ".join(all_plate_coords)
            new_content = (
                f"{instruction} The coordinates of the blocks on the desktop "
                f"are {block_coords_str}, and the coordinates of the plates "
                f"are {plate_coords_str}\n"
            )
            new_updated_content = f"I need to pick up the object located at {objA_2d_real_norm}.\n"

            annotation_with_coords = {
                "0": {
                    "content": new_content,
                    "updated_content": new_updated_content
                }
            }

            self.save_annotation(save_dir, annotation, annotation_with_coords)
            return True

        except Exception as e:
            print(f"Error generating grab data (episode {episode_idx}, idx {idx}): {e}")
            return False


class GrabAndMoveGenerator(DataGeneratorBase):
    """Generator for grab_and_move task."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer,
        phase_generator: PhaseGenerator
    ):
        super().__init__(camera_config, generation_config, instruction_templates, renderer)
        self.phase_gen = phase_generator

    def generate(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        block_fruit_boxes: List[Tuple[np.ndarray, str]],
        plate_boxes: List[Tuple[np.ndarray, str]],
        up_vector: np.ndarray,
        save_dir_base: Path,
        episode_idx: int,
        idx: int
    ) -> bool:
        """Generate grab_and_move data."""
        try:
            method_name = "grab_and_move"
            save_dir = save_dir_base / method_name / f"{idx:02d}"
            (save_dir / "RGB").mkdir(parents=True, exist_ok=True)

            # Select object A and plate B
            box_A, phrase_A = random.choice(block_fruit_boxes)
            objA, objA_2d, objA_2d_real = self.get_jittered_coords(box_A, depth_image)

            box_B, phrase_B = random.choice(plate_boxes)
            objB, objB_2d, objB_2d_real = self.get_jittered_coords(box_B, depth_image)

            # Normalize coordinates
            objA_2d_norm = self.normalize_2d_coords(objA_2d)
            objB_2d_norm = self.normalize_2d_coords(objB_2d)
            objA_2d_real_norm = self.normalize_2d_coords(objA_2d_real)
            objB_2d_real_norm = self.normalize_2d_coords(objB_2d_real)

            # Collect all object coordinates
            all_block_coords = self.collect_object_coords(block_fruit_boxes)
            all_plate_coords = self.collect_object_coords(plate_boxes)

            hand_color = HandColorGenerator.get_random_color()
            step = 0

            # Phase 1: Approach object A
            direction, hand_tip_pos, step, base_vertices = self.phase_gen.generate_phase1(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objA, save_dir, step, episode_idx, idx, method_name, hand_color
            )

            if direction is None:
                return False

            # Phase 2: Move to plate B
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objB, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Create annotations
            instruction = random.choice(self.instructions.grab_and_move)
            annotation = {
                "0": {
                    "content": instruction + "\n",
                    "updated_content": (
                        f"I need to pick up the object located at {objA_2d_norm} "
                        f"and place it on the plate located at {objB_2d_norm}.\n"
                    ),
                }
            }

            block_coords_str = ", ".join(all_block_coords)
            plate_coords_str = ", ".join(all_plate_coords)
            new_content = (
                f"{instruction} The coordinates of the blocks on the desktop "
                f"are {block_coords_str}, and the coordinates of the plates "
                f"are {plate_coords_str}\n"
            )
            new_updated_content = (
                f"I need to pick up the object located at {objA_2d_real_norm} "
                f"and place it on the plate located at {objB_2d_real_norm}.\n"
            )

            annotation_with_coords = {
                "0": {
                    "content": new_content,
                    "updated_content": new_updated_content
                }
            }

            self.save_annotation(save_dir, annotation, annotation_with_coords)
            return True

        except Exception as e:
            print(f"Error generating grab_and_move data (episode {episode_idx}, idx {idx}): {e}")
            return False


class GrabAndMoveTwiceGenerator(DataGeneratorBase):
    """Generator for grab_and_move_twice task."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer,
        phase_generator: PhaseGenerator
    ):
        super().__init__(camera_config, generation_config, instruction_templates, renderer)
        self.phase_gen = phase_generator

    def generate(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        block_fruit_boxes: List[Tuple[np.ndarray, str]],
        plate_boxes: List[Tuple[np.ndarray, str]],
        up_vector: np.ndarray,
        save_dir_base: Path,
        episode_idx: int,
        idx: int
    ) -> bool:
        """Generate grab_and_move_twice data."""
        try:
            method_name = "grab_and_move_twice"
            save_dir = save_dir_base / method_name / f"{idx:02d}"
            (save_dir / "RGB").mkdir(parents=True, exist_ok=True)

            # Need at least 2 blocks
            if len(block_fruit_boxes) < 2:
                return False

            # Select two different blocks and one plate
            box_A, phrase_A = random.choice(block_fruit_boxes)
            objA, objA_2d, objA_2d_real = self.get_jittered_coords(box_A, depth_image)

            remaining_boxes = [b for b in block_fruit_boxes if not np.array_equal(b[0], box_A)]
            if not remaining_boxes:
                return False
            box_B, phrase_B = random.choice(remaining_boxes)
            objB, objB_2d, objB_2d_real = self.get_jittered_coords(box_B, depth_image)

            box_C, phrase_C = random.choice(plate_boxes)
            objC, objC_2d, objC_2d_real = self.get_jittered_coords(box_C, depth_image)

            # Normalize coordinates
            objA_2d_norm = self.normalize_2d_coords(objA_2d)
            objB_2d_norm = self.normalize_2d_coords(objB_2d)
            objC_2d_norm = self.normalize_2d_coords(objC_2d)
            objA_2d_real_norm = self.normalize_2d_coords(objA_2d_real)
            objB_2d_real_norm = self.normalize_2d_coords(objB_2d_real)
            objC_2d_real_norm = self.normalize_2d_coords(objC_2d_real)

            # Collect all object coordinates
            all_block_coords = self.collect_object_coords(block_fruit_boxes)
            all_plate_coords = self.collect_object_coords(plate_boxes)

            hand_color = HandColorGenerator.get_random_color()
            step = 0

            # Phase 1: Approach object A
            direction, hand_tip_pos, step, base_vertices = self.phase_gen.generate_phase1(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objA, save_dir, step, episode_idx, idx, method_name, hand_color
            )

            if direction is None:
                return False

            # Phase 2: Move to object B
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objB, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Phase 3: Move to plate C
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objC, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Create annotations
            instruction = random.choice(self.instructions.grab_and_move_twice)
            annotation = {
                "0": {
                    "content": instruction + "\n",
                    "updated_content": (
                        f"I need to pick up the objects located at {objA_2d_norm} "
                        f"and {objB_2d_norm}, and place them on the plate "
                        f"located at {objC_2d_norm}.\n"
                    ),
                }
            }

            block_coords_str = ", ".join(all_block_coords)
            plate_coords_str = ", ".join(all_plate_coords)
            new_content = (
                f"{instruction} The coordinates of the blocks on the desktop "
                f"are {block_coords_str}, and the coordinates of the plates "
                f"are {plate_coords_str}\n"
            )
            new_updated_content = (
                f"I need to pick up the object located at {objA_2d_real_norm} "
                f"and {objB_2d_real_norm}, and place them on the plate "
                f"located at {objC_2d_real_norm}.\n"
            )

            annotation_with_coords = {
                "0": {
                    "content": new_content,
                    "updated_content": new_updated_content
                }
            }

            self.save_annotation(save_dir, annotation, annotation_with_coords)
            return True

        except Exception as e:
            print(f"Error generating grab_and_move_twice data (episode {episode_idx}, idx {idx}): {e}")
            return False


class GrabAndMoveThriceGenerator(DataGeneratorBase):
    """Generator for grab_and_move_thrice task."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer,
        phase_generator: PhaseGenerator
    ):
        super().__init__(camera_config, generation_config, instruction_templates, renderer)
        self.phase_gen = phase_generator

    def generate(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        block_fruit_boxes: List[Tuple[np.ndarray, str]],
        plate_boxes: List[Tuple[np.ndarray, str]],
        up_vector: np.ndarray,
        save_dir_base: Path,
        episode_idx: int,
        idx: int
    ) -> bool:
        """Generate grab_and_move_thrice data."""
        try:
            method_name = "grab_and_move_thrice"
            save_dir = save_dir_base / method_name / f"{idx:02d}"
            (save_dir / "RGB").mkdir(parents=True, exist_ok=True)

            # Need at least 3 blocks
            if len(block_fruit_boxes) < 3:
                return False

            # Select three different blocks and one plate
            box_A, phrase_A = random.choice(block_fruit_boxes)
            objA, objA_2d, objA_2d_real = self.get_jittered_coords(box_A, depth_image)

            remaining_boxes_1 = [b for b in block_fruit_boxes if not np.array_equal(b[0], box_A)]
            if not remaining_boxes_1:
                return False
            box_B, phrase_B = random.choice(remaining_boxes_1)
            objB, objB_2d, objB_2d_real = self.get_jittered_coords(box_B, depth_image)

            remaining_boxes_2 = [b for b in remaining_boxes_1 if not np.array_equal(b[0], box_B)]
            if not remaining_boxes_2:
                return False
            box_C, phrase_C = random.choice(remaining_boxes_2)
            objC, objC_2d, objC_2d_real = self.get_jittered_coords(box_C, depth_image)

            box_D, phrase_D = random.choice(plate_boxes)
            objD, objD_2d, objD_2d_real = self.get_jittered_coords(box_D, depth_image)

            # Normalize coordinates
            objA_2d_norm = self.normalize_2d_coords(objA_2d)
            objB_2d_norm = self.normalize_2d_coords(objB_2d)
            objC_2d_norm = self.normalize_2d_coords(objC_2d)
            objD_2d_norm = self.normalize_2d_coords(objD_2d)
            objA_2d_real_norm = self.normalize_2d_coords(objA_2d_real)
            objB_2d_real_norm = self.normalize_2d_coords(objB_2d_real)
            objC_2d_real_norm = self.normalize_2d_coords(objC_2d_real)
            objD_2d_real_norm = self.normalize_2d_coords(objD_2d_real)

            # Collect all object coordinates
            all_block_coords = self.collect_object_coords(block_fruit_boxes)
            all_plate_coords = self.collect_object_coords(plate_boxes)

            hand_color = HandColorGenerator.get_random_color()
            step = 0

            # Phase 1: Approach object A
            direction, hand_tip_pos, step, base_vertices = self.phase_gen.generate_phase1(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objA, save_dir, step, episode_idx, idx, method_name, hand_color
            )

            if direction is None:
                return False

            # Phase 2: Move to object B
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objB, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Phase 3: Move to object C
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objC, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Phase 4: Move to plate D
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objD, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Create annotations
            instruction = random.choice(self.instructions.grab_and_move_thrice)
            annotation = {
                "0": {
                    "content": instruction + "\n",
                    "updated_content": (
                        f"I need to pick up the objects located at {objA_2d_norm} "
                        f"and {objB_2d_norm}, and {objC_2d_norm}, and place them on the plate "
                        f"located at {objD_2d_norm}.\n"
                    ),
                }
            }

            block_coords_str = ", ".join(all_block_coords)
            plate_coords_str = ", ".join(all_plate_coords)
            new_content = (
                f"{instruction} The coordinates of the blocks on the desktop "
                f"are {block_coords_str}, and the coordinates of the plates "
                f"are {plate_coords_str}\n"
            )
            new_updated_content = (
                f"I need to pick up the objects located at {objA_2d_real_norm} "
                f"and {objB_2d_real_norm}, and {objC_2d_real_norm}, and place them on the plate "
                f"located at {objD_2d_real_norm}.\n"
            )

            annotation_with_coords = {
                "0": {
                    "content": new_content,
                    "updated_content": new_updated_content
                }
            }

            self.save_annotation(save_dir, annotation, annotation_with_coords)
            return True

        except Exception as e:
            print(f"Error generating grab_and_move_thrice data (episode {episode_idx}, idx {idx}): {e}")
            return False


class GrabOneGenerator(DataGeneratorBase):
    """Generator for grab_one task - select one object from all detected objects."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer,
        phase_generator: PhaseGenerator
    ):
        super().__init__(camera_config, generation_config, instruction_templates, renderer)
        self.phase_gen = phase_generator

    def generate(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        all_boxes: List[Tuple[np.ndarray, str]],
        up_vector: np.ndarray,
        save_dir_base: Path,
        episode_idx: int,
        idx: int
    ) -> bool:
        """Generate grab_one data - pick up one random object."""
        try:
            method_name = "grab_one"
            save_dir = save_dir_base / method_name / f"{idx:02d}"
            (save_dir / "RGB").mkdir(parents=True, exist_ok=True)

            if len(all_boxes) < 1:
                return False

            # Random select one object from all boxes
            box_A, phrase_A = random.choice(all_boxes)
            objA, objA_2d, objA_2d_real = self.get_jittered_coords(box_A, depth_image)

            # Normalize coordinates
            objA_2d_norm = self.normalize_2d_coords(objA_2d)
            objA_2d_real_norm = self.normalize_2d_coords(objA_2d_real)

            # Collect all object coordinates
            all_coords = self.collect_object_coords(all_boxes)

            hand_color = HandColorGenerator.get_random_color()
            step = 0

            # Phase 1: Approach object
            direction, hand_tip_pos, step, _ = self.phase_gen.generate_phase1(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objA, save_dir, step, episode_idx, idx, method_name, hand_color
            )

            if direction is None:
                return False

            # Create annotations
            instruction = random.choice(self.instructions.grab_one)
            annotation = {
                "0": {
                    "content": instruction + "\n",
                    "updated_content": f"I need to pick up the object located at {objA_2d_norm}.\n",
                }
            }

            coords_str = ", ".join(all_coords)
            new_content = (
                f"{instruction} The coordinates of the objects on the desktop "
                f"are {coords_str}\n"
            )
            new_updated_content = f"I need to pick up the object located at {objA_2d_real_norm}.\n"

            annotation_with_coords = {
                "0": {
                    "content": new_content,
                    "updated_content": new_updated_content
                }
            }

            self.save_annotation(save_dir, annotation, annotation_with_coords)
            return True

        except Exception as e:
            print(f"Error generating grab_one data (episode {episode_idx}, idx {idx}): {e}")
            return False


class GrabTwoGenerator(DataGeneratorBase):
    """Generator for grab_two task - select two objects from all detected objects."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer,
        phase_generator: PhaseGenerator
    ):
        super().__init__(camera_config, generation_config, instruction_templates, renderer)
        self.phase_gen = phase_generator

    def generate(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        all_boxes: List[Tuple[np.ndarray, str]],
        up_vector: np.ndarray,
        save_dir_base: Path,
        episode_idx: int,
        idx: int
    ) -> bool:
        """Generate grab_two data - pick up two random objects."""
        try:
            method_name = "grab_two"
            save_dir = save_dir_base / method_name / f"{idx:02d}"
            (save_dir / "RGB").mkdir(parents=True, exist_ok=True)

            # Need at least 2 objects
            if len(all_boxes) < 2:
                return False

            # Select two different objects randomly
            box_A, phrase_A = random.choice(all_boxes)
            objA, objA_2d, objA_2d_real = self.get_jittered_coords(box_A, depth_image)

            remaining_boxes = [b for b in all_boxes if not np.array_equal(b[0], box_A)]
            if not remaining_boxes:
                return False
            box_B, phrase_B = random.choice(remaining_boxes)
            objB, objB_2d, objB_2d_real = self.get_jittered_coords(box_B, depth_image)

            # Normalize coordinates
            objA_2d_norm = self.normalize_2d_coords(objA_2d)
            objB_2d_norm = self.normalize_2d_coords(objB_2d)
            objA_2d_real_norm = self.normalize_2d_coords(objA_2d_real)
            objB_2d_real_norm = self.normalize_2d_coords(objB_2d_real)

            # Collect all object coordinates
            all_coords = self.collect_object_coords(all_boxes)

            hand_color = HandColorGenerator.get_random_color()
            step = 0

            # Phase 1: Approach object A
            direction, hand_tip_pos, step, base_vertices = self.phase_gen.generate_phase1(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objA, save_dir, step, episode_idx, idx, method_name, hand_color
            )

            if direction is None:
                return False

            # Phase 2: Move to object B
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objB, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Create annotations
            instruction = random.choice(self.instructions.grab_two)
            annotation = {
                "0": {
                    "content": instruction + "\n",
                    "updated_content": (
                        f"I need to pick up the objects located at {objA_2d_norm} "
                        f"and {objB_2d_norm}.\n"
                    ),
                }
            }

            coords_str = ", ".join(all_coords)
            new_content = (
                f"{instruction} The coordinates of the objects on the desktop "
                f"are {coords_str}\n"
            )
            new_updated_content = (
                f"I need to pick up the objects located at {objA_2d_real_norm} "
                f"and {objB_2d_real_norm}.\n"
            )

            annotation_with_coords = {
                "0": {
                    "content": new_content,
                    "updated_content": new_updated_content
                }
            }

            self.save_annotation(save_dir, annotation, annotation_with_coords)
            return True

        except Exception as e:
            print(f"Error generating grab_two data (episode {episode_idx}, idx {idx}): {e}")
            return False


class GrabThreeGenerator(DataGeneratorBase):
    """Generator for grab_three task - select three objects from all detected objects."""

    def __init__(
        self,
        camera_config: CameraConfig,
        generation_config: GenerationConfig,
        instruction_templates: InstructionTemplates,
        renderer: HandMeshRenderer,
        phase_generator: PhaseGenerator
    ):
        super().__init__(camera_config, generation_config, instruction_templates, renderer)
        self.phase_gen = phase_generator

    def generate(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        all_boxes: List[Tuple[np.ndarray, str]],
        up_vector: np.ndarray,
        save_dir_base: Path,
        episode_idx: int,
        idx: int
    ) -> bool:
        """Generate grab_three data - pick up three random objects."""
        try:
            method_name = "grab_three"
            save_dir = save_dir_base / method_name / f"{idx:02d}"
            (save_dir / "RGB").mkdir(parents=True, exist_ok=True)

            # Need at least 3 objects
            if len(all_boxes) < 3:
                return False

            # Select three different objects randomly
            box_A, phrase_A = random.choice(all_boxes)
            objA, objA_2d, objA_2d_real = self.get_jittered_coords(box_A, depth_image)

            remaining_boxes_1 = [b for b in all_boxes if not np.array_equal(b[0], box_A)]
            if not remaining_boxes_1:
                return False
            box_B, phrase_B = random.choice(remaining_boxes_1)
            objB, objB_2d, objB_2d_real = self.get_jittered_coords(box_B, depth_image)

            remaining_boxes_2 = [b for b in remaining_boxes_1 if not np.array_equal(b[0], box_B)]
            if not remaining_boxes_2:
                return False
            box_C, phrase_C = random.choice(remaining_boxes_2)
            objC, objC_2d, objC_2d_real = self.get_jittered_coords(box_C, depth_image)

            # Normalize coordinates
            objA_2d_norm = self.normalize_2d_coords(objA_2d)
            objB_2d_norm = self.normalize_2d_coords(objB_2d)
            objC_2d_norm = self.normalize_2d_coords(objC_2d)
            objA_2d_real_norm = self.normalize_2d_coords(objA_2d_real)
            objB_2d_real_norm = self.normalize_2d_coords(objB_2d_real)
            objC_2d_real_norm = self.normalize_2d_coords(objC_2d_real)

            # Collect all object coordinates
            all_coords = self.collect_object_coords(all_boxes)

            hand_color = HandColorGenerator.get_random_color()
            step = 0

            # Phase 1: Approach object A
            direction, hand_tip_pos, step, base_vertices = self.phase_gen.generate_phase1(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objA, save_dir, step, episode_idx, idx, method_name, hand_color
            )

            if direction is None:
                return False

            # Phase 2: Move to object B
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objB, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Phase 3: Move to object C
            direction, hand_tip_pos, step = self.phase_gen.generate_phase2(
                hand_mesh, base_vertices, rgb_image, depth_image,
                objC, direction, hand_tip_pos, up_vector,
                save_dir, step, episode_idx, idx, method_name, hand_color
            )

            # Create annotations
            instruction = random.choice(self.instructions.grab_three)
            annotation = {
                "0": {
                    "content": instruction + "\n",
                    "updated_content": (
                        f"I need to pick up the objects located at {objA_2d_norm}, "
                        f"{objB_2d_norm}, and {objC_2d_norm}.\n"
                    ),
                }
            }

            coords_str = ", ".join(all_coords)
            new_content = (
                f"{instruction} The coordinates of the objects on the desktop "
                f"are {coords_str}\n"
            )
            new_updated_content = (
                f"I need to pick up the objects located at {objA_2d_real_norm}, "
                f"{objB_2d_real_norm}, and {objC_2d_real_norm}.\n"
            )

            annotation_with_coords = {
                "0": {
                    "content": new_content,
                    "updated_content": new_updated_content
                }
            }

            self.save_annotation(save_dir, annotation, annotation_with_coords)
            return True

        except Exception as e:
            print(f"Error generating grab_three data (episode {episode_idx}, idx {idx}): {e}")
            return False
