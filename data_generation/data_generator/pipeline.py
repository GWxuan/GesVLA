"""Main data generation pipeline."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from groundingdino.util.inference import load_model, load_image, predict

from data_generator.configs import CameraConfig, DetectionConfig, GenerationConfig, HandConfig, InstructionTemplates
from data_generator.generators import (
    GrabAndMoveGenerator,
    GrabAndMoveThriceGenerator,
    GrabAndMoveTwiceGenerator,
    GrabGenerator,
    GrabOneGenerator,
    GrabTwoGenerator,
    GrabThreeGenerator,
)
from data_generator.phases import PhaseGenerator
from data_generator.renderer import HandMeshRenderer
from data_generator.utils import CoordinateUtils, HandTransformer, UpVectorCalculator


class DataGenerationPipeline:
    """Main pipeline for data generation."""

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        detection_config: Optional[DetectionConfig] = None,
        hand_config: Optional[HandConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        instruction_templates: Optional[InstructionTemplates] = None
    ):
        """
        Initialize the data generation pipeline.

        Args:
            camera_config: Camera configuration (uses defaults if None)
            detection_config: Detection configuration (uses defaults if None)
            hand_config: Hand configuration (uses defaults if None)
            generation_config: Generation configuration (uses defaults if None)
            instruction_templates: Instruction templates (uses defaults if None)
        """
        self.camera_config = camera_config or CameraConfig()
        self.detection_config = detection_config or DetectionConfig()
        self.hand_config = hand_config or HandConfig()
        self.gen_config = generation_config or GenerationConfig()
        self.instructions = instruction_templates or InstructionTemplates()

        # Initialize renderer
        self.renderer = HandMeshRenderer(self.camera_config)

        # Initialize phase generator
        self.phase_gen = PhaseGenerator(self.camera_config, self.renderer, self.gen_config)

        # Initialize task generators
        self.grab_gen = GrabGenerator(
            self.camera_config, self.gen_config, self.instructions,
            self.renderer, self.phase_gen
        )
        self.grab_and_move_gen = GrabAndMoveGenerator(
            self.camera_config, self.gen_config, self.instructions,
            self.renderer, self.phase_gen
        )
        self.grab_and_move_twice_gen = GrabAndMoveTwiceGenerator(
            self.camera_config, self.gen_config, self.instructions,
            self.renderer, self.phase_gen
        )
        self.grab_and_move_thrice_gen = GrabAndMoveThriceGenerator(
            self.camera_config, self.gen_config, self.instructions,
            self.renderer, self.phase_gen
        )
        self.grab_one_gen = GrabOneGenerator(
            self.camera_config, self.gen_config, self.instructions,
            self.renderer, self.phase_gen
        )
        self.grab_two_gen = GrabTwoGenerator(
            self.camera_config, self.gen_config, self.instructions,
            self.renderer, self.phase_gen
        )
        self.grab_three_gen = GrabThreeGenerator(
            self.camera_config, self.gen_config, self.instructions,
            self.renderer, self.phase_gen
        )

        # Up vector calculator
        self.up_vector_calc = UpVectorCalculator(self.camera_config)

        # Coordinate utilities
        self.coord_utils = CoordinateUtils(self.camera_config)

        # Detection model (loaded lazily)
        self._detection_model = None

    def _load_detection_model(self):
        """Load the detection model."""
        if self._detection_model is None:
            self._detection_model = load_model(
                self.detection_config.model_config_path,
                self.detection_config.model_weights_path
            )
        return self._detection_model

    def _detect_objects(
        self,
        rgb_path: Path
    ) -> Tuple[List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]], List[Tuple[np.ndarray, str]]]:
        """
        Detect objects in the image.

        Args:
            rgb_path: Path to RGB image

        Returns:
            Tuple of (block_boxes, plate_boxes, all_boxes)
        """
        model = self._load_detection_model()

        _, image = load_image(str(rgb_path))

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=self.detection_config.text_prompt,
            box_threshold=self.detection_config.box_threshold,
            text_threshold=self.detection_config.text_threshold
        )

        block_fruit_boxes = []
        plate_boxes = []
        all_boxes = []

        for box, _, phrase in zip(boxes, logits, phrases):
            all_boxes.append((box, phrase))
            if "colored block" in phrase or "block" in phrase:
                block_fruit_boxes.append((box, phrase))
            elif "yellow plate" in phrase or "pink plate" in phrase or "plate" in phrase:
                plate_boxes.append((box, phrase))

        return block_fruit_boxes, plate_boxes, all_boxes


    @staticmethod
    def get_next_idx(save_dir: Path, method_name: str) -> int:
        """
        Get the next available index for a method.

        Args:
            save_dir: Save directory
            method_name: Method name

        Returns:
            Next available index
        """
        method_dir = save_dir / method_name
        if not method_dir.exists():
            return 0

        max_idx = -1
        for item in method_dir.iterdir():
            if item.is_dir():
                try:
                    idx = int(item.name)
                    max_idx = max(max_idx, idx)
                except ValueError:
                    continue
        return max_idx + 1

    def _get_transform_configs(self) -> List[Tuple[np.ndarray, Callable, str, str]]:
        """
        Get transformation configurations for both hands.

        Returns:
            List of (pre_move, transform_func, mesh_path, group_id) tuples
        """
        return [
            (
                self.hand_config.right_hand_pre_move,
                HandTransformer.pre_transform_right,
                self.hand_config.right_hand_mesh_path,
                "right_hand"
            ),
            (
                self.hand_config.left_hand_pre_move,
                HandTransformer.pre_transform_left,
                self.hand_config.left_hand_mesh_path,
                "left_hand"
            )
        ]

    def run(self) -> None:
        """Run the data generation pipeline."""
        # Initialize renderer
        self.renderer.initialize()

        try:
            # Get episode directories
            data_root = Path(self.gen_config.data_root)
            episode_dirs = sorted([
                d for d in data_root.iterdir()
                if d.is_dir() and d.name.startswith("episode_")
            ])

            # Transform configurations
            transform_configs = self._get_transform_configs()

            # Progress bar for episodes
            desc = f"episodes {self.gen_config.start_episode}-{self.gen_config.end_episode}"
            with tqdm(total=len(episode_dirs), desc=desc) as pbar_episodes:
                for episode_dir in episode_dirs:
                    episode_idx = int(episode_dir.name.split("_")[1])

                    # Skip episodes outside range
                    if not (
                        self.gen_config.start_episode <= episode_idx <= self.gen_config.end_episode
                    ):
                        pbar_episodes.update(1)
                        continue

                    try:
                        print(f"\nProcessing {episode_dir.name}...")

                        # Load images
                        rgb_path = episode_dir / "right_rgb_frame_0.png"
                        depth_path = episode_dir / "depth_frame_0.png"
                        rgb_image = cv2.imread(str(rgb_path))
                        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)

                        # Detect objects
                        block_fruit_boxes, plate_boxes, all_boxes = self._detect_objects(rgb_path)
                        
                        # Check which task types are enabled
                        enabled_tasks = self.gen_config.get_enabled_tasks()
                        original_tasks = ["grab", "grab_and_move", "grab_and_move_twice", "grab_and_move_thrice"]
                        flexible_tasks = ["grab_one", "grab_two", "grab_three"]
                        
                        has_original_tasks = any(t[0] in original_tasks for t in enabled_tasks)
                        has_flexible_tasks = any(t[0] in flexible_tasks for t in enabled_tasks)
                        
                        # Skip logic based on enabled task types
                        if has_original_tasks and (len(block_fruit_boxes) == 0 or len(plate_boxes) == 0):
                            if not has_flexible_tasks:
                                print(f"Could not find blocks/plates in {episode_dir.name}, skipping...")
                                pbar_episodes.update(1)
                                continue
                            else:
                                # Only run flexible tasks if we have objects
                                if len(all_boxes) == 0:
                                    print(f"No objects detected in {episode_dir.name}, skipping...")
                                    pbar_episodes.update(1)
                                    continue
                                print(f"No blocks/plates found, only running flexible tasks for {episode_dir.name}...")
                        elif has_flexible_tasks and len(all_boxes) == 0:
                            print(f"No objects detected in {episode_dir.name}, skipping...")
                            pbar_episodes.update(1)
                            continue

                        # Convert color channel
                        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

                        # Calculate up vector
                        up_vector = self.up_vector_calc.calculate(block_fruit_boxes, depth_image)

                        # Process each hand configuration
                        for pre_move, transform_func, hand_mesh_path, group_id in transform_configs:
                            # Load hand mesh
                            hand_mesh = o3d.io.read_triangle_mesh(hand_mesh_path)

                            # Calculate base vertices
                            base_vertices = np.asarray(hand_mesh.vertices) * self.hand_config.scale * 0.13
                            base_vertices = transform_func(base_vertices, pre_move)

                            # Create output directory
                            generate_dir = Path(
                                f"{self.gen_config.output_root}/episode_{episode_idx}_{group_id}"
                            )
                            generate_dir.mkdir(parents=True, exist_ok=True)
                            print(f"\nGenerating {group_id} data to {generate_dir}...")

                            # Build method configurations dynamically from enabled tasks
                            task_to_generator = {
                                "grab": self.grab_gen,
                                "grab_and_move": self.grab_and_move_gen,
                                "grab_and_move_twice": self.grab_and_move_twice_gen,
                                "grab_and_move_thrice": self.grab_and_move_thrice_gen,
                                "grab_one": self.grab_one_gen,
                                "grab_two": self.grab_two_gen,
                                "grab_three": self.grab_three_gen,
                            }
                            
                            # Get enabled tasks from config
                            enabled_tasks = self.gen_config.get_enabled_tasks()
                            methods = [
                                (task_name, task_to_generator[task_name], num_samples)
                                for task_name, num_samples in enabled_tasks
                            ]
                            
                            # all_boxes is already prepared from detection

                            # Filter methods based on available objects
                            can_run_original = len(block_fruit_boxes) > 0 and len(plate_boxes) > 0
                            filtered_methods = []
                            for method_name, generator, num_samples in methods:
                                if method_name in original_tasks:
                                    if can_run_original:
                                        filtered_methods.append((method_name, generator, num_samples))
                                else:  # flexible tasks
                                    # Check if we have enough objects for the task
                                    required_objects = {"grab_one": 1, "grab_two": 2, "grab_three": 3}
                                    if len(all_boxes) >= required_objects.get(method_name, 1):
                                        filtered_methods.append((method_name, generator, num_samples))

                            # Generate data for each method
                            for method_name, generator, num_samples in filtered_methods:
                                start_idx = self.get_next_idx(generate_dir, method_name)

                                print(f"\nGenerating {method_name} data for {episode_dir.name}...")
                                idx = start_idx
                                max_attempts = num_samples * self.gen_config.max_attempts_multiplier
                                attempts = 0

                                with tqdm(total=num_samples, desc=f"{method_name}") as pbar_method:
                                    while idx < num_samples + start_idx and attempts < max_attempts:
                                        attempts += 1

                                        # New flexible tasks use all_boxes
                                        if method_name in ["grab_one", "grab_two", "grab_three"]:
                                            success = generator.generate(
                                                hand_mesh,
                                                base_vertices.copy(),
                                                rgb_image,
                                                depth_image,
                                                all_boxes,
                                                up_vector,
                                                generate_dir,
                                                episode_idx,
                                                idx
                                            )
                                        # Original grab task (no up_vector)
                                        elif method_name == "grab":
                                            success = generator.generate(
                                                hand_mesh,
                                                base_vertices.copy(),
                                                rgb_image,
                                                depth_image,
                                                block_fruit_boxes,
                                                plate_boxes,
                                                generate_dir,
                                                episode_idx,
                                                idx
                                            )
                                        # Original move tasks (with up_vector)
                                        else:
                                            success = generator.generate(
                                                hand_mesh,
                                                base_vertices.copy(),
                                                rgb_image,
                                                depth_image,
                                                block_fruit_boxes,
                                                plate_boxes,
                                                up_vector,
                                                generate_dir,
                                                episode_idx,
                                                idx
                                            )

                                        if success:
                                            idx += 1
                                            pbar_method.update(1)

                        pbar_episodes.update(1)

                    except Exception as e:
                        print(f"Error processing episode {episode_dir.name}: {e}")
                        pbar_episodes.update(1)

        except Exception as e:
            print(f"Error in main pipeline: {e}")
        finally:
            # Clean up renderer
            self.renderer.cleanup()
