"""MediaPipe keypoint detection for train data."""

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)


def clean_detect_directory(detect_dir: Path) -> bool:
    """Clean detect directory (images + keypoint.json)."""
    if detect_dir.exists():
        for png_file in detect_dir.glob("*.png"):
            png_file.unlink()
        for jpg_file in detect_dir.glob("*.jpg"):
            jpg_file.unlink()
        keypoint_file = detect_dir / "keypoint.json"
        if keypoint_file.exists():
            keypoint_file.unlink()
        return True
    return False


def detect_hand_keypoints(rgb_dir: Path) -> Tuple[bool, str]:
    """Detect keypoints from frames containing 'start' or 'end'."""
    all_frames = [f for f in rgb_dir.glob("*.png") if "start" in f.name or "end" in f.name]

    def extract_number(filename: Path) -> int:
        """Extract the first number found in a filename."""
        import re
        numbers = re.findall(r"\d+", filename.name)
        return int(numbers[0]) if numbers else 0

    selected_frames = sorted(all_frames, key=extract_number)
    if not selected_frames:
        return False, "No frames with 'start' or 'end' found"

    detect_dir = rgb_dir.parent / "detect"
    detect_dir.mkdir(parents=True, exist_ok=True)
    clean_detect_directory(detect_dir)

    keypoint_data = []

    for i, frame_path in enumerate(selected_frames):
        image = cv2.imread(str(frame_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        point_feature = [0.0] * 12
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoint_mapping = [
                    (8, 0, 1, 2),
                    (7, 3, 4, 5),
                    (6, 6, 7, 8),
                    (0, 9, 10, 11)
                ]
                for landmark_idx, x_idx, y_idx, z_idx in keypoint_mapping:
                    if 0 <= landmark_idx < len(hand_landmarks.landmark):
                        landmark = hand_landmarks.landmark[landmark_idx]
                        point_feature[x_idx] = landmark.x
                        point_feature[y_idx] = landmark.y
                        point_feature[z_idx] = landmark.z

                        x_pixel = int(landmark.x * image.shape[1])
                        y_pixel = int(landmark.y * image.shape[0])
                        cv2.circle(image, (x_pixel, y_pixel), 3, (0, 255, 0), -1)

        frame_filename = f"frame_{i:04d}.jpg"
        cv2.imwrite(str(detect_dir / frame_filename), image)

        keypoint_data.append({
            "image": frame_filename,
            "point_feature": point_feature,
        })

    with open(detect_dir / "keypoint.json", "w") as f:
        json.dump(keypoint_data, f, indent=4)

    return True, "Success"


def collect_all_directories(base_path: Path) -> List[Tuple[str, str, str, Path]]:
    """Collect all RGB directories under train data."""
    directories = []

    for episode_dir in sorted(base_path.iterdir()):
        if (
            episode_dir.is_dir()
            and episode_dir.name.startswith("episode_")
            and ("_right_hand" in episode_dir.name or "_left_hand" in episode_dir.name)
        ):
            for method_dir in sorted(episode_dir.glob("*")):
                if not method_dir.is_dir() or method_dir.name.startswith("."):
                    continue
                for idx_dir in sorted(method_dir.glob("*")):
                    if not idx_dir.is_dir():
                        continue
                    rgb_dir = idx_dir / "RGB"
                    if rgb_dir.exists():
                        directories.append((episode_dir.name, method_dir.name, idx_dir.name, rgb_dir))

    return directories


def run_detect(base_path: str) -> None:
    """Run MediaPipe detection for all train data directories."""
    base = Path(base_path)

    print("Scanning directory structure...")
    directories = collect_all_directories(base)
    total_dirs = len(directories)
    print(f"Found {total_dirs} data directories to process")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    with tqdm(total=total_dirs, desc="Processing data directories") as pbar:
        for episode_name, method_name, idx_name, rgb_dir in directories:
            pbar.set_postfix({
                "episode": episode_name,
                "method": method_name,
                "idx": idx_name,
            })
            try:
                success, _ = detect_hand_keypoints(rgb_dir)
                if success:
                    processed_count += 1
                else:
                    skipped_count += 1
            except Exception as exc:
                error_count += 1
                _ = f"Error: {exc}"
            pbar.update(1)

    print("\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
