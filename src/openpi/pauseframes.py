import cv2
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import json
import random
import argparse

# MediaPipe hand detection
import mediapipe as mp
mp_hands = mp.solutions.hands

# Camera intrinsics
width, height = 1280, 720

# Instruction templates
content_grab_and_move = [
    "Instruction: pick this up and put it there.",
    "Instruction: take this one and place it on that.",
    "Instruction: pick it up and put it over there.",
    "Instruction: carry this and release it there.",
    "Instruction: stack this block over there.",
    "Instruction: move this onto that one.",
]

DIFF_THRESH = 20
MOTION_RATIO_THRESH = 0.01
MIN_STATIC_LEN = 6

def apply_mask_to_frame(frame):
    """
    Apply a mask to the frame by filling the region left of the line
    connecting (345, 0) and (170, 720) with zeros to avoid detecting the operator's hand.
    """
    mask_polygon = np.array([
        [0, 0],
        [345, 0],
        [170, 720],
        [0, 720]
    ], dtype=np.int32)
    
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    cv2.fillPoly(mask, [mask_polygon], 0)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    return masked_frame

def clean_detect_directory(detect_dir):
    """Clear png/json files in detect_dir while keeping thought.json."""
    if detect_dir.exists():
        for png_file in detect_dir.glob("*.png"):
            png_file.unlink()
        for jpg_file in detect_dir.glob("*.jpg"):
            jpg_file.unlink()
        
        keypoint_file = detect_dir / "keypoint.json"
        if keypoint_file.exists():
            keypoint_file.unlink()

        selection_file = detect_dir / "selection_info.json"
        if selection_file.exists():
            selection_file.unlink()
        
        return True
    return False

def detect_activity_start_from_image_changes(frames):
    """
    Detect activity start based on image changes.
    Returns: frame index for activity start based on image changes.
    """
    if len(frames) < 5:
        return 0
    
    diffs = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_bin = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(diff_bin)
        motion_ratio = non_zero / diff_bin.size
        diffs.append(motion_ratio)
        prev_gray = gray
    
    diffs.insert(0, 0)
    
    activity_start_from_image = None
    consecutive_active = 0
    
    for i in range(len(diffs)):
        if diffs[i] > 0.05:
            consecutive_active += 1
            if consecutive_active >= 2:
                activity_start_from_image = max(0, i - 2 + 1)
                break
        else:
            consecutive_active = 0
    
    if activity_start_from_image is None:
        activity_start_from_image = min(5, len(frames) - 1)
    
    return activity_start_from_image

def process_video_with_mediapipe(video_path):
    """
    Process a video in one pass: read all frames, detect hand keypoints, and
    determine the hand interval.
    Returns: (all_frames, all_keypoints, valid_mask, hand_start, hand_end)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return None, None, None, None, None
    
    frames = []
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    n_frames = len(frames)
    cap = cv2.VideoCapture(str(video_path))
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.2,
        model_complexity=1
    )
    
    all_frames = []
    all_keypoints = np.zeros((n_frames, 12), dtype=float)
    valid_mask = np.zeros(n_frames, dtype=bool)
    hand_detected_frames = []
    
    print(f"正在处理视频: {video_path.name}")
    
    for i in tqdm(range(n_frames), desc="帧处理"):
        ret, frame = cap.read()
        if not ret:
            break
        
        masked_frame = apply_mask_to_frame(frame)
        all_frames.append(masked_frame)
        
        img_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        hand_detected = results.multi_hand_landmarks is not None
        if hand_detected:
            hand_detected_frames.append(i)
            valid_mask[i] = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            mapping = [
                (8, 0, 1, 2),   # Index fingertip
                (7, 3, 4, 5),   # Index first joint
                (6, 6, 7, 8),   # Index second joint
                (0, 9, 10, 11)  # Wrist
            ]
            
            for lm_idx, x_idx, y_idx, z_idx in mapping:
                lm = hand_landmarks.landmark[lm_idx]
                all_keypoints[i, x_idx] = lm.x
                all_keypoints[i, y_idx] = lm.y
                all_keypoints[i, z_idx] = lm.z
    
    cap.release()
    hands.close()
    
    if not hand_detected_frames:
        print(f"警告: 在视频 {video_path.name} 中未检测到手部")
        return all_frames, all_keypoints, valid_mask, None, None
    
    hand_start = max(0, hand_detected_frames[0])  
    hand_end = hand_detected_frames[-1]
    
    if hand_end - hand_start < MIN_STATIC_LEN * 2:
        print(f"警告: 手部区间太短 ({hand_end - hand_start} 帧)")
        return all_frames, all_keypoints, valid_mask, hand_start, hand_end
    
    return all_frames, all_keypoints, valid_mask, hand_start, hand_end

def interpolate_keypoints(all_keypoints, valid_mask):
    """Interpolate keypoints across frames."""
    n_frames = len(all_keypoints)
    keypoints_interp = np.zeros_like(all_keypoints)
    
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) >= 2:
        first_valid = valid_idx[0]
        keypoints_interp[:first_valid] = 0.0
        
        for j in range(12):
            col = all_keypoints[:, j]
            keypoints_interp[first_valid:, j] = np.interp(
                np.arange(first_valid, n_frames),
                valid_idx,
                col[valid_idx]
            )
    else:
        print("警告: 有效关键点帧数不足，跳过插值")
        keypoints_interp = all_keypoints.copy()
    
    return keypoints_interp

def extract_pause_frames_in_interval(all_frames, hand_start, hand_end, num_pause_frames=2):
    """
    在手部区间内使用帧间差异方法检测停顿帧
    
    参数:
        all_frames: 所有视频帧
        hand_start: 手部区间开始帧
        hand_end: end frame of the hand interval
        num_pause_frames: number of pause frames to extract (1, 2, or 3)

    Returns: list of pause frame indices
    """
    if hand_start is None or hand_end is None:
        return []
    
    interval_frames = all_frames[hand_start:hand_end+1]
    
    if len(interval_frames) < MIN_STATIC_LEN * 2:
        print(f"警告: 手部区间帧数不足 ({len(interval_frames)} 帧)")
        return []
    
    # Detect all static segments.
    static_segments = []
    current_static_start = None
    prev_gray = cv2.cvtColor(interval_frames[0], cv2.COLOR_BGR2GRAY)
    
    for i in range(1, len(interval_frames)):
        frame = interval_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_bin = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(diff_bin)
        motion_ratio = non_zero / diff_bin.size
        
        if motion_ratio < MOTION_RATIO_THRESH:
            if current_static_start is None:
                current_static_start = i
        else:
            if current_static_start is not None:
                seg_len = i - current_static_start
                if seg_len >= MIN_STATIC_LEN:
                    static_segments.append((
                        current_static_start + hand_start, 
                        i - 1 + hand_start
                    ))
                current_static_start = None
        
        prev_gray = gray
    
    # If the video ends while still static.
    if current_static_start is not None:
        seg_len = len(interval_frames) - current_static_start
        if seg_len >= MIN_STATIC_LEN:
            static_segments.append((
                current_static_start + hand_start, 
                hand_end
            ))
    
    print(f"检测到 {len(static_segments)} 个静止段: {static_segments}")
    
    # Select frames based on the requested count.
    pause_frames = _select_pause_frames(static_segments, hand_start, hand_end, num_pause_frames)
    
    # Ensure pause frames lie within the hand interval.
    pause_frames = [max(hand_start, min(p, hand_end)) for p in pause_frames]
    
    return pause_frames

def _select_pause_frames(static_segments, hand_start, hand_end, num_pause_frames):
    """
    Select pause frames based on static segments and target count.

    Strategy:
    - 1 frame: if segments exist, use the longest segment center; otherwise hand_end-10
    - 2 frames: if >=2 segments, use centers of first and last;
         if 1 segment, use its center and hand_end-10;
         if 0 segments, use hand_start+5 and hand_end-10
    - 3 frames: if >=3 segments, use centers of first, second, and last;
         if 2 segments, use first center, mid between segments, and last center;
         if 1 segment, use 1/4, 1/2, 3/4 positions;
         if 0 segments, distribute evenly between hand_start and hand_end
    """
    num_segments = len(static_segments)
    
    def get_segment_center(seg):
        """Get the center frame of a static segment."""
        return (seg[0] + seg[1]) // 2
    
    def get_segment_length(seg):
        """Get the length of a static segment."""
        return seg[1] - seg[0] + 1
    
    if num_pause_frames == 1:
        # === Select 1 frame ===
        if num_segments >= 1:
            # Use the center of the longest static segment.
            longest_seg = max(static_segments, key=get_segment_length)
            return [get_segment_center(longest_seg)]
        else:
            # No static segment; use hand_end-10.
            return [max(hand_start, hand_end - 10)]
    
    elif num_pause_frames == 2:
        # === Select 2 frames ===
        if num_segments >= 2:
            # Use centers of the first and last static segments.
            first_mid = get_segment_center(static_segments[0])
            last_mid = get_segment_center(static_segments[-1])
            if last_mid - first_mid >= 5:
                return [first_mid, last_mid]
            else:
                return [first_mid, hand_end - 10]
        elif num_segments == 1:
            # Only one segment: use its center and hand_end-10.
            mid = get_segment_center(static_segments[0])
            return [mid, max(mid + 10, hand_end - 10)]
        else:
            # No static segments.
            return [hand_start + 5, max(hand_start + 15, hand_end - 10)]
    
    elif num_pause_frames == 3:
        # === Select 3 frames ===
        if num_segments >= 3:
            # Use centers of the first, second, and last static segments.
            first_mid = get_segment_center(static_segments[0])
            second_mid = get_segment_center(static_segments[1])
            last_mid = get_segment_center(static_segments[-1])
            return [first_mid, second_mid, last_mid]
        elif num_segments == 2:
            # Two segments: first center, mid between segments, last center.
            first_mid = get_segment_center(static_segments[0])
            second_mid = get_segment_center(static_segments[-1])
            last = hand_end - 10
            return [first_mid, second_mid, last]
        # elif num_segments == 1:
        #     # 一个静止段：选该段的1/4、1/2、3/4位置
        #     seg = static_segments[0]
        #     seg_len = seg[1] - seg[0]
        #     if seg_len >= 6:  # 段足够长，内部取点
        #         q1 = seg[0] + seg_len // 4
        #         q2 = seg[0] + seg_len // 2
        #         q3 = seg[0] + 3 * seg_len // 4
        #         return [q1, q2, q3]
        #     else:
        #         # 段太短，在整个区间均匀分布
        #         total_len = hand_end - hand_start
        #         return [
        #             hand_start + total_len // 4,
        #             hand_start + total_len // 2,
        #             hand_start + 3 * total_len // 4
        #         ]
        else:
            # No static segments; distribute evenly.
            total_len = hand_end - hand_start
            return [
                hand_start + 5,
                hand_start + total_len // 2,
                hand_end - 10
            ]
    
    else:
        raise ValueError(f"不支持的停顿帧数量: {num_pause_frames}，仅支持 1, 2, 3")

def generate_thought_json(detect_dir):
    """Generate thought.json."""
    try:
        instruction = random.choice(content_grab_and_move)
        
        content = f"{instruction}"
        updated_content = f"blank"
        
        thought_data = {
            "0": {
                "content": content,
                "updated_content": updated_content,
                "updated_content_w_instruction": content,
            }
        }
        
        thought_file = detect_dir / "thought.json"
        with open(thought_file, "w") as f:
            json.dump(thought_data, f, indent=4)
        
        print(f"已生成thought.json文件: {thought_file}")
        return True
        
    except Exception as e:
        print(f"生成thought.json失败: {e}")
        return False

def process_single_video(video_path, detect_dir, num_pause_frames=2):
    """
    Main function to process a single video.

    Args:
        video_path: video file path
        detect_dir: output directory
        num_pause_frames: number of pause frames to extract (1, 2, or 3)
    """
    video_path = Path(video_path)
    detect_dir = Path(detect_dir)
    detect_dir.mkdir(parents=True, exist_ok=True)
    
    clean_detect_directory(detect_dir)
    
    print(f"\n处理视频: {video_path.name}")
    
    # Step 1: process video.
    all_frames, all_keypoints, valid_mask, hand_start, hand_end = process_video_with_mediapipe(video_path)

    print(f"最终手部检测区间: 第 {hand_start} 帧到第 {hand_end} 帧")
    
    if all_frames is None:
        return False, "视频读取失败"
    
    if hand_start is None or hand_end is None:
        return False, "未检测到手部"
    
    # Step 2: interpolate keypoints.
    keypoints_interp = interpolate_keypoints(all_keypoints, valid_mask)
    
    # Step 3: detect pause frames within the hand interval (configurable count).
    pause_frames = extract_pause_frames_in_interval(all_frames, hand_start, hand_end, num_pause_frames)
    
    if len(pause_frames) < num_pause_frames:
        print(f"警告: 未检测到足够的停顿帧，当前: {len(pause_frames)}, 需要: {num_pause_frames}")
        # Fill in missing frames.
        while len(pause_frames) < num_pause_frames:
            if len(pause_frames) == 0:
                pause_frames.append(hand_start + 5)
            else:
                # Add after the last frame.
                next_frame = min(pause_frames[-1] + 20, hand_end - 5)
                if next_frame not in pause_frames:
                    pause_frames.append(next_frame)
                else:
                    pause_frames.append(hand_end)
                    break
    
    print(f"选定的停顿帧索引: {pause_frames}")
    
    # Step 4: generate thought.json.
    generate_thought_json(detect_dir)
    
    # Step 5: save pause frames and keypoints.
    keypoint_data = []
    
    for i, frame_idx in enumerate(pause_frames):
        if frame_idx >= len(all_frames):
            print(f"警告: 帧索引 {frame_idx} 超出范围")
            continue
        
        frame = all_frames[frame_idx].copy()
        original_frame = all_frames[frame_idx].copy()
        
        original_fname = f"original_frame_pause_{i}.jpg"
        cv2.imwrite(str(detect_dir / original_fname), original_frame)
        
        kp = keypoints_interp[frame_idx]
        
        for k in range(0, 12, 3):
            if kp[k] != 0 or kp[k+1] != 0:
                x = int(kp[k] * frame.shape[1])
                y = int(kp[k+1] * frame.shape[0])
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        fname = f"frame_pause_{i}.jpg"
        cv2.imwrite(str(detect_dir / fname), frame)
        
        keypoint_data.append({
            "frame_index": int(frame_idx),
            "image": fname,
            "point_feature": kp.tolist(),
            "is_valid": bool(valid_mask[frame_idx])
        })
    
    with open(detect_dir / "keypoint.json", "w") as f:
        json.dump(keypoint_data, f, indent=4)

    print(f"处理完成: 保存了 {len(pause_frames)} 个停顿帧")
    return True, "成功"

def collect_all_videos(base_path, start_episode=None, end_episode=None):
    """
    Collect all videos to be processed.

    Args:
        base_path: dataset root directory
        start_episode: starting episode number (inclusive), None to start from the beginning
        end_episode: ending episode number (inclusive), None to go to the end

    Returns: [(episode_name, video_path, detect_dir), ...]
    """
    videos = []
    base_path = Path(base_path)
    
    def get_episode_num(episode_dir):
        """Extract the numeric index from an episode directory name."""
        try:
            return int(episode_dir.name.split('_')[1])
        except:
            return -1
    
    # Traverse and sort all episode directories.
    all_episodes = sorted(base_path.glob("episode_*"), key=get_episode_num)
    
    for episode_dir in all_episodes:
        if not episode_dir.is_dir():
            continue
        
        episode_num = get_episode_num(episode_dir)
        if episode_num < 0:
            continue
        
        # Check whether it is within the specified range.
        if start_episode is not None and episode_num < start_episode:
            continue
        if end_episode is not None and episode_num > end_episode:
            continue
            
        # Check whether the video file exists.
        video_path = episode_dir / "gesture_right_realsense_rgb.mp4"
        if video_path.exists():
            videos.append((episode_dir.name, video_path, episode_dir / "detect"))
    
    return videos

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description='从视频中提取停顿帧')
    parser.add_argument('--base_path', type=str,
                        default="data/datasets/pick_block",
                        help='数据集根目录')
    parser.add_argument('--num_pause_frames', type=int, default=2, choices=[1, 2, 3],
                        help='需要提取的停顿帧数量 (1, 2, 或 3)')
    parser.add_argument('--start_episode', type=int, default=None,
                        help='起始episode编号（包含）')
    parser.add_argument('--end_episode', type=int, default=None,
                        help='结束episode编号（包含）')
    
    args = parser.parse_args()
    
    base_path = args.base_path
    num_pause_frames = args.num_pause_frames
    start_episode = args.start_episode
    end_episode = args.end_episode
    
    print(f"配置:")
    print(f"  - 数据集路径: {base_path}")
    print(f"  - 停顿帧数量: {num_pause_frames}")
    print(f"  - Episode范围: {start_episode if start_episode else '开始'} -> {end_episode if end_episode else '结束'}")
    print()
    
    # Collect videos to process.
    print("正在扫描视频文件...")
    videos = collect_all_videos(base_path, start_episode, end_episode)
    total_videos = len(videos)
    
    print(f"找到 {total_videos} 个视频需要处理")
    
    if total_videos == 0:
        print("没有找到需要处理的视频，请检查路径和episode范围")
        return
    
    # Track processing results.
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Process all videos with a global progress bar.
    with tqdm(total=total_videos, desc="处理视频") as pbar:
        for episode_name, video_path, detect_dir in videos:
            pbar.set_postfix({'episode': episode_name})
            
            try:
                success, message = process_single_video(video_path, detect_dir, num_pause_frames)
                if success:
                    processed_count += 1
                else:
                    skipped_count += 1
                    print(f"跳过 {episode_name}: {message}")
            except Exception as e:
                error_count += 1
                print(f"处理 {episode_name} 时出错: {str(e)}")
            
            pbar.update(1)
    
    # Output final summary.
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count}")
    print(f"跳过: {skipped_count}")
    print(f"错误: {error_count}")

if __name__ == "__main__":
    main()
