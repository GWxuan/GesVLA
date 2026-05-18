"""Hand mesh rendering utilities."""

from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d

from data_generator.configs import CameraConfig
from data_generator.utils import HandColorGenerator


class HandMeshRenderer:
    """Manages Open3D offscreen rendering for hand meshes."""

    def __init__(self, camera_config: CameraConfig):
        """
        Initialize the renderer.

        Args:
            camera_config: Camera intrinsic parameters
        """
        self.camera = camera_config
        self.renderer: Optional[o3d.visualization.rendering.OffscreenRenderer] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the offscreen renderer."""
        if self._initialized:
            return

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(
            self.camera.width, self.camera.height
        )
        self.renderer.scene.set_background([0.63, 0.53, 0.50, 1])

        # Set up camera
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.camera.width, self.camera.height,
            self.camera.fx, self.camera.fy,
            self.camera.cx, self.camera.cy
        )
        extrinsic = np.eye(4)
        self.renderer.setup_camera(intrinsic, extrinsic)

        K = self.camera.get_intrinsic_matrix()
        self.renderer.scene.camera.set_projection(
            K, 0.01, 10.0, self.camera.width, self.camera.height
        )

        # Set up lighting
        scn = self.renderer.scene.scene
        try:
            scn.enable_sun_light(False)
        except Exception:
            pass
        try:
            scn.set_indirect_light_intensity(17000.0)
        except Exception:
            pass

        self._initialized = True

    def render(
        self,
        mesh: o3d.geometry.TriangleMesh,
        hand_color: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render a hand mesh to color and depth images.

        Args:
            mesh: Open3D triangle mesh of the hand
            hand_color: RGBA color for the hand (optional)

        Returns:
            Tuple of (color_image, depth_image)
        """
        if not self._initialized:
            self.initialize()

        # Remove previous geometry
        try:
            self.renderer.scene.remove_geometry("hand")
        except Exception:
            pass

        # Create material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"

        # Set hand color
        if hand_color is None:
            hand_color = HandColorGenerator.get_random_color()

        if mesh.has_vertex_colors():
            mat.base_color = [1.0, 1.0, 1.0, 1.0]
        else:
            mat.base_color = hand_color

        self.renderer.scene.add_geometry("hand", mesh, mat)

        # Render
        color = self.renderer.render_to_image()
        depth = self.renderer.render_to_depth_image(z_in_view_space=True)

        color = np.asarray(color)
        depth = np.nan_to_num(np.asarray(depth), nan=0.0, posinf=0.0, neginf=0.0)
        depth = (depth * 1000).astype(np.uint16)

        return color, depth

    def cleanup(self) -> None:
        """Clean up renderer resources."""
        if self.renderer is not None:
            try:
                self.renderer.scene.clear_geometry()
                self.renderer = None
                self._initialized = False
            except Exception as e:
                print(f"Error cleaning up renderer: {e}")
