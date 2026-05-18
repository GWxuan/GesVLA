"""Utility classes and math helpers for data generation."""

import random
from typing import List, Tuple

import numpy as np
import trimesh

from data_generator.configs import CameraConfig


class MathUtils:
    """Mathematical utility functions."""

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            return v
        return v / n

    @staticmethod
    def rotation_from_vecs(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix that rotates v_from to v_to.

        Args:
            v_from: Source direction vector
            v_to: Target direction vector

        Returns:
            3x3 rotation matrix
        """
        v_from = MathUtils.normalize(v_from)
        v_to = MathUtils.normalize(v_to)
        axis = np.cross(v_from, v_to)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-9:
            # Vectors are parallel or anti-parallel
            if np.dot(v_from, v_to) < 0:
                # 180 degree rotation: choose arbitrary perpendicular axis
                if abs(v_from[0]) < 0.9:
                    tmp = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
                else:
                    tmp = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
                tmp = MathUtils.normalize(tmp)
                R4 = trimesh.transformations.rotation_matrix(np.pi, tmp)
                return R4[:3, :3]
            return np.eye(3)

        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
        R4 = trimesh.transformations.rotation_matrix(angle, axis)
        return R4[:3, :3]


class CoordinateUtils:
    """Coordinate transformation utilities."""

    def __init__(self, camera_config: CameraConfig):
        """
        Initialize with camera configuration.

        Args:
            camera_config: Camera intrinsic parameters
        """
        self.camera = camera_config

    def pixel_to_3d(self, x: int, y: int, depth: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to 3D world coordinates.

        Args:
            x: Pixel x coordinate
            y: Pixel y coordinate
            depth: Depth image

        Returns:
            3D point in camera coordinates
        """
        z = depth[y, x] * self.camera.depth_scale
        x_3d = (x - self.camera.cx) * z / self.camera.fx
        y_3d = (y - self.camera.cy) * z / self.camera.fy
        return np.array([x_3d, y_3d, z])

    def project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project 3D point to 2D pixel coordinates.

        Args:
            point_3d: 3D point in camera coordinates

        Returns:
            Tuple of (u, v) pixel coordinates
        """
        x, y, z = point_3d
        u = int(x * self.camera.fx / z + self.camera.cx)
        v = int(y * self.camera.fy / z + self.camera.cy)
        return u, v


class PlaneUtils:
    """Plane fitting utilities."""

    @staticmethod
    def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a plane to points using SVD.

        Args:
            points: Nx3 array of 3D points

        Returns:
            Tuple of (normal vector, centroid)
        """
        centroid = points.mean(axis=0)
        X = points - centroid
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        normal = vh[-1, :]
        normal /= np.linalg.norm(normal)
        return normal, centroid

    @staticmethod
    def fit_plane_ransac(
        points: np.ndarray,
        thresh: float = 0.01,
        max_iter: int = 300
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a plane to points using RANSAC.

        Args:
            points: Nx3 array of 3D points
            thresh: Inlier distance threshold
            max_iter: Maximum iterations

        Returns:
            Tuple of (normal vector, centroid)
        """
        best_inliers, best_model = [], None
        N = points.shape[0]

        for _ in range(max_iter):
            ids = np.random.choice(N, 3, replace=False)
            normal, centroid = PlaneUtils.fit_plane_svd(points[ids])
            dists = np.abs((points - centroid) @ normal)
            inliers = np.where(dists < thresh)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = (normal, centroid)
                if len(best_inliers) > 0.8 * N:
                    break

        if best_model is None:
            return PlaneUtils.fit_plane_svd(points)

        inlier_pts = points[best_inliers]
        return PlaneUtils.fit_plane_svd(inlier_pts)


class HandColorGenerator:
    """Generate realistic hand skin colors."""

    # Predefined skin color palette
    PREDEFINED_COLORS: List[List[float]] = [
        [221 / 255, 158 / 255, 108 / 255, 1.0],  # Original
        [200 / 255, 140 / 255, 100 / 255, 1.0],  # Slightly darker
        [240 / 255, 180 / 255, 140 / 255, 1.0],  # Slightly lighter
        [180 / 255, 120 / 255, 80 / 255, 1.0],  # Darker
        [255 / 255, 200 / 255, 160 / 255, 1.0],  # Lighter
        [120 / 255, 80 / 255, 60 / 255, 1.0],  # Deep brown
        [80 / 255, 50 / 255, 30 / 255, 1.0],  # Dark brown
    ]

    @staticmethod
    def get_random_color() -> List[float]:
        """
        Generate a random skin color within specified range.

        Returns:
            RGBA color list with values in [0, 1]
        """
        skin_r = random.uniform(0.91, 0.95)
        skin_g = random.uniform(0.65, 0.69)
        skin_b = random.uniform(0.47, 0.50)
        return [skin_r, skin_g, skin_b, 1.0]

    @classmethod
    def get_predefined_color(cls) -> List[float]:
        """
        Get a random color from predefined palette.

        Returns:
            RGBA color list
        """
        return random.choice(cls.PREDEFINED_COLORS)


class HandTransformer:
    """Handles hand mesh transformations."""

    @staticmethod
    def pre_transform_right(vertices: np.ndarray, pre_move: np.ndarray) -> np.ndarray:
        """
        Apply pre-transformation for right hand.

        Args:
            vertices: Hand mesh vertices
            pre_move: Pre-transformation offset

        Returns:
            Transformed vertices
        """
        vertices = vertices + pre_move
        initial_dir = np.array([-1.0, -1.0, -1.0])
        initial_dir /= np.linalg.norm(initial_dir)
        target_dir = np.array([0.0, 0.0, -1.0])
        target_dir /= np.linalg.norm(target_dir)

        axis = np.cross(initial_dir, target_dir)
        if np.linalg.norm(axis) < 1e-6:
            dot = np.dot(initial_dir, target_dir)
            if dot < 0:
                axis = (
                    np.cross(initial_dir, [1, 0, 0])
                    if abs(initial_dir[0]) < 0.9
                    else np.cross(initial_dir, [0, 1, 0])
                )
                axis /= np.linalg.norm(axis)
                angle = np.pi
            else:
                return vertices
        else:
            axis /= np.linalg.norm(axis)
            dot = np.clip(np.dot(initial_dir, target_dir), -1.0, 1.0)
            angle = np.arccos(dot)

        R = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
        return np.dot(vertices, R.T)

    @staticmethod
    def pre_transform_left(vertices: np.ndarray, pre_move: np.ndarray) -> np.ndarray:
        """
        Apply pre-transformation for left hand.

        Args:
            vertices: Hand mesh vertices
            pre_move: Pre-transformation offset

        Returns:
            Transformed vertices
        """
        vertices = vertices + pre_move
        initial_dir = np.array([-1.0, 1.0, 1.0])
        initial_dir /= np.linalg.norm(initial_dir)
        target_dir = np.array([0.0, 0.0, 1.0])
        target_dir /= np.linalg.norm(target_dir)

        axis = np.cross(initial_dir, target_dir)
        if np.linalg.norm(axis) < 1e-6:
            dot = np.dot(initial_dir, target_dir)
            if dot < 0:
                axis = (
                    np.cross(initial_dir, [1, 0, 0])
                    if abs(initial_dir[0]) < 0.9
                    else np.cross(initial_dir, [0, 1, 0])
                )
                axis /= np.linalg.norm(axis)
                angle = np.pi
            else:
                return vertices
        else:
            axis /= np.linalg.norm(axis)
            dot = np.clip(np.dot(initial_dir, target_dir), -1.0, 1.0)
            angle = np.arccos(dot)

        R = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]
        transformed_points = np.dot(vertices, R.T)

        # Additional rotation around Z-axis
        z_rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
        z_rotation_matrix = z_rotation_matrix[:3, :3]
        transformed_points = np.dot(transformed_points, z_rotation_matrix.T)

        return transformed_points

    @staticmethod
    def main_transform(
        vertices: np.ndarray,
        direction: np.ndarray,
        hand_tip_pos: np.ndarray
    ) -> np.ndarray:
        """
        Apply main transformation to orient hand in specified direction.

        Args:
            vertices: Hand mesh vertices
            direction: Target pointing direction
            hand_tip_pos: Position of hand tip

        Returns:
            Transformed vertices
        """
        initial = np.array([0.0, 0.0, 1.0])
        target = direction / np.linalg.norm(direction)
        axis = np.cross(initial, target)

        if np.linalg.norm(axis) < 1e-6:
            if np.dot(initial, target) < 0:
                axis = np.array([0, 1, 0])
            else:
                return vertices

        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.dot(initial, target))
        R = trimesh.transformations.rotation_matrix(angle, axis)[:3, :3]

        return np.dot(vertices, R.T) + hand_tip_pos

    @staticmethod
    def is_hand_visible(
        vertices: np.ndarray,
        camera: CameraConfig,
        threshold: float = 0.4
    ) -> bool:
        """
        Check if hand is sufficiently visible in camera view.

        Args:
            vertices: Hand mesh vertices
            camera: Camera configuration
            threshold: Minimum visible vertex ratio

        Returns:
            True if hand is visible enough
        """
        K = camera.get_intrinsic_matrix()
        proj = np.dot(vertices, K.T)
        proj = proj / proj[:, 2:3]
        u, v = proj[:, 0], proj[:, 1]

        visible_ratio = np.mean(
            (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        )
        return visible_ratio > threshold


class UpVectorCalculator:
    """Calculate the up direction vector from detected objects."""

    def __init__(self, camera_config: CameraConfig):
        """
        Initialize with camera configuration.

        Args:
            camera_config: Camera intrinsic parameters
        """
        self.camera = camera_config
        self.coord_utils = CoordinateUtils(camera_config)

    def ensure_up_direction(self, normal: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
        """
        Ensure the up vector points upward in image space.

        Args:
            normal: Plane normal vector
            reference_point: Reference point in 3D

        Returns:
            Corrected up vector
        """
        eps = 0.01
        p0 = reference_point

        u0, v0 = self.coord_utils.project_3d_to_2d(p0)
        u_up, v_up = self.coord_utils.project_3d_to_2d(p0 + normal * eps)
        u_down, v_down = self.coord_utils.project_3d_to_2d(p0 - normal * eps)

        if v_up < v0:
            return normal / np.linalg.norm(normal)
        if v_down < v0:
            return -normal / np.linalg.norm(normal)
        return -normal if normal[1] > 0 else normal

    def calculate(
        self,
        block_fruit_boxes: List[Tuple[np.ndarray, str]],
        depth_image: np.ndarray
    ) -> np.ndarray:
        """
        Calculate up vector from detected object positions.

        Args:
            block_fruit_boxes: List of (box, phrase) tuples
            depth_image: Depth image

        Returns:
            Up direction vector
        """
        if len(block_fruit_boxes) >= 3:
            block_pts = []
            for box, _ in block_fruit_boxes:
                center_x_norm, center_y_norm = box[0], box[1]

                x = int(center_x_norm * self.camera.width)
                y = int(center_y_norm * self.camera.height)
                x = np.clip(x, 0, self.camera.width - 1)
                y = np.clip(y, 0, self.camera.height - 1)

                pt3 = self.coord_utils.pixel_to_3d(x, y, depth_image)
                if pt3 is not None:
                    block_pts.append(pt3)

            block_pts = np.array(block_pts)
            if block_pts.shape[0] >= 3:
                plane_normal, plane_centroid = PlaneUtils.fit_plane_ransac(block_pts)
                up_vector = self.ensure_up_direction(plane_normal, plane_centroid)
                return up_vector

        # Default up direction
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
