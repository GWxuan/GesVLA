"""Phase generation utilities for hand motion."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import random
import trimesh

from data_generator.configs import CameraConfig, GenerationConfig
from data_generator.renderer import HandMeshRenderer
from data_generator.utils import HandTransformer, MathUtils


class PhaseGenerator:
    """Generates different phases of hand movement."""

    def __init__(
        self,
        camera_config: CameraConfig,
        renderer: HandMeshRenderer,
        generation_config: Optional[GenerationConfig] = None
    ):
        """
        Initialize the phase generator.

        Args:
            camera_config: Camera configuration
            renderer: Hand mesh renderer
        """
        self.camera = camera_config
        self.renderer = renderer
        self.transformer = HandTransformer()
        self.gen_config = generation_config or GenerationConfig()
        self.enable_lift = bool(self.gen_config.enable_lift)
        self.video_data = bool(self.gen_config.video_data)

    def generate_phase1(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        target_pos: np.ndarray,
        save_dir: Path,
        step: int,
        episode_idx: int,
        idx: int,
        method_name: str,
        hand_color: Optional[List[float]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int, np.ndarray]:
        """
        Generate phase 1: Approach the target object from a distance.

        Args:
            hand_mesh: Hand mesh object
            base_vertices: Base vertices of hand mesh
            rgb_image: RGB image
            depth_image: Depth image
            target_pos: Target position (objA)
            save_dir: Save directory
            step: Current step number
            episode_idx: Episode index
            idx: Sample index
            method_name: Method name
            hand_color: Hand color

        Returns:
            Tuple of (direction, hand_tip_pos, step, base_vertices)
        """
        # Random approach direction
        direction = np.array([
            random.uniform(-0.7, -0.2),
            random.uniform(-0.8, 0.8),
            random.uniform(0.1, 0.7)
        ])
        direction /= np.linalg.norm(direction)

        if direction[1] < 0:
            # Rotate base vertices around z-axis by 60 degrees
            angle = np.pi / 3
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            base_vertices = np.dot(base_vertices, rotation_matrix.T)

        step_size = 0.002

        # Find maximum step
        max_step = 0
        base_vertices_tmp = self.transformer.main_transform(
            base_vertices.copy(), direction, target_pos - direction * 0.04
        )
        if not self.transformer.is_hand_visible(base_vertices_tmp, self.camera):
            return None, None, step, base_vertices

        while True:
            distance = 0.04 + max_step * step_size
            hand_tip_pos = target_pos - direction * distance
            u = int(hand_tip_pos[0] * self.camera.fx / hand_tip_pos[2] + self.camera.cx)
            v = int(hand_tip_pos[1] * self.camera.fy / hand_tip_pos[2] + self.camera.cy)
            if not (
                0 <= u < self.camera.width
                and 0 <= v < self.camera.height
                and hand_tip_pos[2] > 0
            ):
                break
            max_step += 1

        threshold = random.uniform(0.005, 0.01)

        last_composed = None
        # Approach from farthest point
        for step_idx in range(max_step, -1, -1):
            distance = threshold + step_idx * step_size
            hand_tip_pos = target_pos - direction * distance
            if self.video_data:
                vertices = self.transformer.main_transform(
                    base_vertices.copy(), direction, hand_tip_pos
                )
                hand_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                hand_rgb, hand_depth = self.renderer.render(hand_mesh, hand_color)
                hand_rgb_bgr = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2BGR)
                composed_rgb = rgb_image.copy()
                mask = (hand_depth > 0)
                composed_rgb[mask] = hand_rgb_bgr[mask]
                cv2.imwrite(str(save_dir / "RGB" / f"frame_{step:05d}.png"), composed_rgb)
                last_composed = composed_rgb
            step += 1

        vertices = self.transformer.main_transform(
            base_vertices.copy(), direction, hand_tip_pos
        )
        hand_mesh.vertices = o3d.utility.Vector3dVector(vertices)

        hand_rgb, hand_depth = self.renderer.render(hand_mesh, hand_color)
        #hand_rgb_bgr = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2BGR)

        composed_rgb = rgb_image.copy()
        mask = (hand_depth > 0)
        composed_rgb[mask] = hand_rgb[mask]

        cv2.imwrite(str(save_dir / "RGB" / f"frame_{step:05d}_start.png"), cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR))
        step += 1
        last_composed = cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR)

        for _ in range(10):
            if self.video_data and last_composed is not None:
                cv2.imwrite(str(save_dir / "RGB" / f"frame_{step:05d}.png"), last_composed)
            step += 1

        return direction, hand_tip_pos, step, base_vertices

    def generate_phase2(
        self,
        hand_mesh: o3d.geometry.TriangleMesh,
        base_vertices: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        target_pos: np.ndarray,
        direction: np.ndarray,
        hand_tip_pos: np.ndarray,
        up_vector: np.ndarray,
        save_dir: Path,
        step: int,
        episode_idx: int,
        idx: int,
        method_name: str,
        hand_color: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Generate phase 2: Move from current position to target with parabolic lift.

        Args:
            hand_mesh: Hand mesh object
            base_vertices: Base vertices of hand mesh
            rgb_image: RGB image
            depth_image: Depth image
            target_pos: Target position (objB)
            direction: Current direction
            hand_tip_pos: Current hand tip position
            up_vector: Up direction vector
            save_dir: Save directory
            step: Current step number
            episode_idx: Episode index
            idx: Sample index
            method_name: Method name
            hand_color: Hand color

        Returns:
            Tuple of (direction, hand_tip_pos, step)
        """
        tip_current = hand_tip_pos.copy()
        dir_current = direction.copy()

        # Select local pivot point (elbow)
        pivot_local_idx = np.argmin(base_vertices[:, 2])
        pivot_local = base_vertices[pivot_local_idx].copy()

        # Calculate initial elbow position
        R_main = MathUtils.rotation_from_vecs(np.array([0.0, 0.0, 1.0]), dir_current)
        elbow_current = (R_main @ pivot_local) + tip_current

        # Control parameters
        max_angle_per_frame = np.deg2rad(0.5)
        move_step = 0.001
        min_distance = random.uniform(0.005, 0.01)
        angle_threshold = np.deg2rad(2)

        if self.enable_lift:
            # Pre-compute actual total distance
            tip_precompute = tip_current.copy()
            dir_precompute = dir_current.copy()
            elbow_precompute = elbow_current.copy()
            actual_total_dist = 0.0
            prev_tip_precompute = tip_precompute.copy()

            step_count_precompute = 0
            max_steps_precompute = 200

            while step_count_precompute < max_steps_precompute:
                to_B = target_pos - tip_precompute
                dist_to_B = np.linalg.norm(to_B)
                if dist_to_B < 1e-6:
                    break
                dir_to_B = to_B / dist_to_B
                angle = np.arccos(np.clip(np.dot(dir_precompute, dir_to_B), -1.0, 1.0))
                if dist_to_B <= min_distance + 1e-3 and angle < np.deg2rad(1):
                    break
                if angle < np.deg2rad(1):
                    break

                # Pre-compute rotation + translation
                if angle > np.deg2rad(1):
                    rot_axis = np.cross(dir_precompute, dir_to_B)
                    if dist_to_B < min_distance:
                        dir_next_precompute = dir_precompute.copy()
                        tip_rotated_precompute = tip_precompute.copy()
                    elif np.linalg.norm(rot_axis) > 1e-6:
                        rot_axis /= np.linalg.norm(rot_axis)
                        step_angle = min(max_angle_per_frame, angle)
                        R = trimesh.transformations.rotation_matrix(step_angle, rot_axis)[:3, :3]
                        dir_next_precompute = (R @ dir_precompute)
                        dir_next_precompute /= np.linalg.norm(dir_next_precompute)
                        tip_to_elbow = tip_precompute - elbow_precompute
                        tip_rotated_precompute = elbow_precompute + (R @ tip_to_elbow)
                    else:
                        dir_next_precompute = dir_precompute.copy()
                        tip_rotated_precompute = tip_precompute.copy()
                else:
                    dir_next_precompute = dir_precompute.copy()
                    tip_rotated_precompute = tip_precompute.copy()

                target_tip = target_pos - dir_next_precompute * min_distance
                move_dir = target_tip - tip_rotated_precompute
                move_dist = np.linalg.norm(move_dir)
                if move_dist > 0:
                    move_dir /= move_dist
                    tip_next_precompute = tip_rotated_precompute + move_dir * min(move_step, move_dist)
                    elbow_next_precompute = elbow_precompute + move_dir * min(move_step, move_dist)
                else:
                    tip_next_precompute = tip_rotated_precompute.copy()
                    elbow_next_precompute = elbow_precompute.copy()

                # Accumulate actual distance
                step_dist_precompute = np.linalg.norm(tip_next_precompute - prev_tip_precompute)
                actual_total_dist += step_dist_precompute
                prev_tip_precompute = tip_next_precompute.copy()

                tip_precompute = tip_next_precompute
                dir_precompute = dir_next_precompute
                elbow_precompute = elbow_next_precompute
                step_count_precompute += 1

            # Re-initialize parameters with actual total distance
            total_dist = actual_total_dist
            peak_shift = 0.5
            h_max = random.uniform(0.0, 0.02)
            h_max = min(h_max, total_dist * 0.5)
        else:
            total_dist = 0.0
            peak_shift = 0.5
            h_max = 0.0

        # Formal rendering process
        cumulative_dist = 0.0
        prev_tip = tip_current.copy()
        last_composed = None

        step_count = 0
        max_steps = 200
        angle_threshold = np.deg2rad(1)

        dir_next = dir_current.copy()
        tip_next_vis = tip_current.copy()

        while step_count < max_steps:
            to_B = target_pos - tip_current
            dist_to_B = np.linalg.norm(to_B)
            if dist_to_B < 1e-6:
                break
            dir_to_B = to_B / dist_to_B
            angle = np.arccos(np.clip(np.dot(dir_current, dir_to_B), -1.0, 1.0))
            if dist_to_B <= min_distance + 1e-3 and angle < angle_threshold:
                break
            if angle < angle_threshold:
                break

            # Base rotation + translation
            if angle > angle_threshold:
                rot_axis = np.cross(dir_current, dir_to_B)
                if dist_to_B < min_distance:
                    dir_next = dir_current.copy()
                    tip_rotated = tip_current.copy()
                elif np.linalg.norm(rot_axis) > 1e-6:
                    rot_axis /= np.linalg.norm(rot_axis)
                    step_angle = min(max_angle_per_frame, angle)
                    R = trimesh.transformations.rotation_matrix(step_angle, rot_axis)[:3, :3]
                    dir_next = (R @ dir_current)
                    dir_next /= np.linalg.norm(dir_next)
                    tip_to_elbow = tip_current - elbow_current
                    tip_rotated = elbow_current + (R @ tip_to_elbow)
                else:
                    dir_next = dir_current.copy()
                    tip_rotated = tip_current.copy()
            else:
                dir_next = dir_current.copy()
                tip_rotated = tip_current.copy()

            target_tip = target_pos - dir_next * min_distance
            move_dir = target_tip - tip_rotated
            move_dist = np.linalg.norm(move_dir)
            if move_dist > 0:
                move_dir /= move_dist
                tip_next = tip_rotated + move_dir * min(move_step, move_dist)
                elbow_next = elbow_current + move_dir * min(move_step, move_dist)
            else:
                tip_next = tip_rotated.copy()
                elbow_next = elbow_current.copy()

            # Cumulative path progress
            step_dist = np.linalg.norm(tip_next - prev_tip)
            cumulative_dist += step_dist
            prev_tip = tip_next.copy()
            progress = np.clip(cumulative_dist / (total_dist + 1e-6), 0.0, 1.0)

            if self.enable_lift:
                # Parabolic lift
                norm_p = (progress - peak_shift) / 0.5
                z_offset = h_max * max(0, 1 - norm_p ** 2)
                tip_next_vis = tip_next + up_vector * z_offset
            else:
                tip_next_vis = tip_next

            # Update state
            tip_current = tip_next
            dir_current = dir_next
            elbow_current = elbow_next
            step += 1
            step_count += 1

            if self.video_data:
                verts = self.transformer.main_transform(
                    base_vertices.copy(), dir_next, tip_next_vis
                )
                hand_mesh.vertices = o3d.utility.Vector3dVector(verts)
                hand_rgb, hand_depth = self.renderer.render(hand_mesh, hand_color)
                #hand_rgb_bgr = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2BGR)
                composed_rgb = rgb_image.copy()
                mask = (hand_depth > 0)
                composed_rgb[mask] = hand_rgb[mask]
                cv2.imwrite(str(save_dir / "RGB" / f"frame_{step:05d}.png"), cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR))
                last_composed = cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR)

        verts = self.transformer.main_transform(base_vertices.copy(), dir_next, tip_next_vis)
        hand_mesh.vertices = o3d.utility.Vector3dVector(verts)
        hand_rgb, hand_depth = self.renderer.render(hand_mesh, hand_color)
        #hand_rgb_bgr = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2BGR)
        composed_rgb = rgb_image.copy()
        mask = (hand_depth > 0)
        composed_rgb[mask] = hand_rgb[mask]
        last_composed = composed_rgb

        # End frame
        cv2.imwrite(str(save_dir / "RGB" / f"frame_{step:05d}_end.png"), cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR))
        step += 1

        for _ in range(10):
            if self.video_data and last_composed is not None:
                cv2.imwrite(str(save_dir / "RGB" / f"frame_{step:05d}.png"), last_composed)
            step += 1

        return dir_current, tip_current, step
