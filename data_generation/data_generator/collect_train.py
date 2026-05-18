"""Collect train data into parquet and reasoning files."""

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj


def _collect_directories(base_path: Path) -> List[Tuple[Path, Path, Path, Path]]:
    """Collect (episode, method, index, detect) directories to process."""
    directories = []

    for episode_dir in sorted(base_path.iterdir()):
        if (
            episode_dir.is_dir()
            and episode_dir.name.startswith("episode_")
            and ("_right_hand" in episode_dir.name or "_left_hand" in episode_dir.name)
        ):
            method_dirs = [d for d in episode_dir.iterdir() if d.is_dir()]
            if not method_dirs:
                continue

            for method_dir in sorted(method_dirs):
                for sub_dir in sorted(method_dir.glob("*")):
                    detect_dir = sub_dir / "detect"
                    if detect_dir.exists():
                        directories.append((episode_dir, method_dir, sub_dir, detect_dir))

    return directories


def run_collect(base_path: str) -> None:
    """Collect train data into parquet + reasoning files."""
    expected_frames = {
        "grab": 1,
        "grab_and_move": 2,
        "grab_and_move_twice": 3,
        "grab_and_move_thrice": 4,
        "grab_one": 1,
        "grab_two": 2,
        "grab_three": 3,
    }

    root = Path(base_path)

    all_records = []
    reasoning: Dict[str, List] = {"vision_language_episode_idx": []}
    reasoning_with_coordinate: Dict[str, List] = {"vision_language_episode_idx": []}

    processed_count = 0
    skipped_count = 0
    global_index = 0

    print("Scanning directory structure...")
    directories = _collect_directories(root)
    total_dirs = len(directories)
    print(f"Found {total_dirs} data directories to process")

    with tqdm(total=total_dirs, desc="Processing data directories") as pbar:
        for episode_dir, method_dir, sub_dir, detect_dir in directories:
            pbar.set_postfix({
                "episode": episode_dir.name,
                "task": method_dir.name,
                "idx": sub_dir.name,
            })

            thought_path = detect_dir / "thought.json"
            kp_path = detect_dir / "keypoint.json"
            new_thought_path = detect_dir / "thought_with_coordinate.json"

            if not thought_path.exists() or not kp_path.exists() or not new_thought_path.exists():
                skipped_count += 1
                pbar.update(1)
                continue

            try:
                with open(thought_path, "r") as f:
                    tdata = json.load(f)["0"]
                content = tdata["content"].strip()
                updated = tdata["updated_content"].strip()
                updated_w_instr = f"{content}\n{updated}"

                with open(new_thought_path, "r") as f:
                    coord_tdata = json.load(f)["0"]
                coord_content = coord_tdata["content"].strip()
                coord_updated = coord_tdata["updated_content"].strip()
                coord_updated_w_instr = f"{coord_content}\n{coord_updated}"

                origin_idx = int(episode_dir.name.split("_")[1])

                reasoning[str(global_index)] = {
                    "segments": [
                        {
                            "content": content,
                            "updated_content": updated,
                        }
                    ]
                }
                reasoning["vision_language_episode_idx"].append(global_index)

                reasoning_with_coordinate[str(global_index)] = {
                    "segments": [
                        {
                            "content": coord_content,
                            "updated_content": coord_updated,
                        }
                    ]
                }
                reasoning_with_coordinate["vision_language_episode_idx"].append(global_index)

                with open(kp_path, "r") as f:
                    kps = json.load(f)

                kp_map = {item["image"]: item["point_feature"] for item in kps}

                frame_files = sorted([f for f in os.listdir(detect_dir) if f.endswith(".jpg")])
                method_name = method_dir.name
                if method_name not in expected_frames:
                    skipped_count += 1
                    pbar.update(1)
                    continue

                if len(frame_files) != expected_frames[method_name]:
                    print(
                        f"Skipping {detect_dir}: {method_name} requires {expected_frames[method_name]} frames, "
                        f"but found {len(frame_files)}"
                    )
                    skipped_count += 1
                    pbar.update(1)
                    continue

                images_bytes = []
                point_features = []

                for fname in frame_files:
                    fpath = detect_dir / fname

                    with Image.open(fpath) as img:
                        img = img.convert("RGB")
                        img_buffer = io.BytesIO()
                        img.save(img_buffer, format="JPEG", quality=95)
                        images_bytes.append(img_buffer.getvalue())

                    if fname in kp_map:
                        point_features.append(kp_map[fname])
                    else:
                        point_features.append([0.0] * 12)

                episode_data = {
                    "image": images_bytes,
                    "episode_index": global_index,
                    "task_index": global_index,
                    "origin_episode_idx": origin_idx,
                    "point_feature": point_features,
                }

                all_records.append(episode_data)
                processed_count += 1
                global_index += 1

            except Exception as e:
                skipped_count += 1
                print(f"Error processing {detect_dir}: {e}")

            pbar.update(1)

    output_dir = root / "pointing_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    if all_records:
        schema = pa.schema([
            ("image", pa.list_(pa.binary())),
            ("point_feature", pa.list_(pa.list_(pa.float32()))),
            ("origin_episode_idx", pa.int64()),
            ("episode_index", pa.int64()),
            ("task_index", pa.int64()),
        ])

        table = pa.Table.from_pylist(all_records, schema=schema)
        parquet_path = output_dir / "data.parquet"
        pq.write_table(table, parquet_path)

        print(f"✅ Dataset saved to {parquet_path}")
    else:
        print("⚠️ No records to save")

    if reasoning["vision_language_episode_idx"]:
        reasoning_path = output_dir / "reasoning.json"
        with open(reasoning_path, "w", encoding="utf-8") as f:
            json.dump(reasoning, f, indent=2, ensure_ascii=False)
        print(f"✅ reasoning.json saved to {reasoning_path}")
    else:
        print("⚠️ No reasoning data to save")

    if reasoning_with_coordinate["vision_language_episode_idx"]:
        coord_reasoning_path = output_dir / "reasoning_with_coordinate.json"
        with open(coord_reasoning_path, "w", encoding="utf-8") as f:
            json.dump(reasoning_with_coordinate, f, indent=2, ensure_ascii=False)
        print(f"✅ reasoning_with_coordinate.json saved to {coord_reasoning_path}")
    else:
        print("⚠️ No reasoning_with_coordinate data to save")

    print("\n=== Processing summary ===")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    if processed_count + skipped_count > 0:
        print(f"Completion rate: {processed_count/(processed_count+skipped_count)*100:.1f}%")
    print("===================")
