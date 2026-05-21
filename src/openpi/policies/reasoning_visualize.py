"""Utilities to visualize reasoning coordinates on images."""

from pathlib import Path
import re
from typing import Iterable

import cv2
import numpy as np


def resize_with_pad_reverse(
    normalized_coords: tuple[float, float],
    original_height: int,
    original_width: int,
    target_size: int = 224,
) -> tuple[int, int]:
    """
    Map normalized coordinates from a padded 224×224 image back to the original image.

    Args:
        normalized_coords: Normalized (x, y) in [0, 1] for the 224×224 padded image.
        original_height: Original image height.
        original_width: Original image width.
        target_size: Target size used for padding (default: 224).

    Returns:
        (x, y) in the original image coordinate system.
    """
    # Compute resize ratio used to fit the original image into target_size.
    ratio = max(original_width / target_size, original_height / target_size)

    # Compute resized dimensions.
    resized_height = int(original_height / ratio)
    resized_width = int(original_width / ratio)

    # Compute padding applied during resize_with_pad.
    pad_h0, _ = divmod(target_size - resized_height, 2)
    pad_w0, _ = divmod(target_size - resized_width, 2)

    # Convert normalized coords back to 224×224 pixels.
    x_224 = normalized_coords[0] * target_size
    y_224 = normalized_coords[1] * target_size

    # Remove padding.
    x_resized = x_224 - pad_w0
    y_resized = y_224 - pad_h0

    # Scale back to original size.
    x_original = x_resized * ratio
    y_original = y_resized * ratio

    return int(x_original), int(y_original)


def extract_coordinates_from_text(text: str) -> list[tuple[float, float]]:
    """
    Extract (x, y) coordinate pairs from text.

    Supported formats: (x,y) or (x, y)
    """
    pattern = r"\(([0-9.]+),([0-9.]+)\)"
    matches = re.findall(pattern, text)
    return [(float(x), float(y)) for x, y in matches]


def _draw_points_on_image(
    image: np.ndarray,
    coordinates: Iterable[tuple[float, float]],
    *,
    max_points: int | None = None,
    colors: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Draw coordinate points on a copy of the image and return it."""
    output = image.copy()
    height, width = output.shape[:2]

    if colors is None:
        colors = [
            (0, 0, 255),    # Red (BGR)
            (0, 255, 255),  # Yellow (BGR)
            (0, 255, 0),    # Green (BGR)
        ]

    for i, (x_norm, y_norm) in enumerate(coordinates):
        if max_points is not None and i >= max_points:
            break

        x_orig, y_orig = resize_with_pad_reverse(
            (x_norm, y_norm),
            original_height=height,
            original_width=width,
        )

        # Clamp to image bounds.
        x_orig = max(0, min(x_orig, width - 1))
        y_orig = max(0, min(y_orig, height - 1))

        color = colors[i % len(colors)]

        cv2.circle(output, (x_orig, y_orig), 10, color, -1)
        cv2.circle(output, (x_orig, y_orig), 15, (255, 255, 255), 2)

        label = f"Point {i + 1} ({x_norm:.3f}, {y_norm:.3f})"
        cv2.putText(
            output,
            label,
            (x_orig + 20, y_orig - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return output


def visualprompt(image: np.ndarray, reasoning_text: str) -> np.ndarray | None:
    """
    Draw reasoning coordinates on an in-memory image and return the result.

    Args:
        image: Input image (RGB or BGR). The function does not change color space.
        reasoning_text: Text containing (x, y) coordinates.

    Returns:
        Image with overlaid points, or None if no coordinates are found.
    """
    coordinates = extract_coordinates_from_text(reasoning_text)
    if not coordinates:
        return None

    return _draw_points_on_image(image, coordinates, max_points=3)


def draw_coordinates_on_image(
    image_path: str,
    coordinates: list[tuple[float, float]],
    output_path: str | None = None,
) -> None:
    """Draw coordinates on an image file and save the result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    if output_path is None:
        input_path = Path(image_path)
        output_path = str(input_path.parent / f"visualized_{input_path.name}")

    result = _draw_points_on_image(image, coordinates, max_points=None)
    cv2.imwrite(output_path, result)
    print(f"Saved visualization: {output_path}")


def draw_coordinates_on_image_multiframe(
    image_path: str,
    coordinates: list[tuple[float, float]],
    output_path: str | None = None,
) -> None:
    """
    Draw up to 3 coordinates on an image.

    Color scheme:
        1st point: red
        2nd point: yellow
        3rd point: green
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    if output_path is None:
        input_path = Path(image_path)
        output_path = str(input_path.parent / f"visualized_{input_path.name}")

    result = _draw_points_on_image(image, coordinates, max_points=3)
    cv2.imwrite(output_path, result)
    print(f"Saved visualization: {output_path}")


def visualize(reasoning_text: str, image_path: str) -> None:
    """Parse coordinates from text and visualize them on an image file."""
    coordinates = extract_coordinates_from_text(reasoning_text)

    print(f"Found {len(coordinates)} coordinates:")
    for i, coord in enumerate(coordinates):
        print(f"Point {i + 1}: {coord}")

    draw_coordinates_on_image_multiframe(image_path, coordinates)
