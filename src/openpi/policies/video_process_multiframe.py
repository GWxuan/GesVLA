"""
Multi-pause-frame video processing module.
Supports a configurable number of pause frames (1, 2, or 3).
Keeps the same interface as video_process.py and adds the frame_num parameter.
"""

import cv2
import numpy as np
from tqdm import tqdm

# MediaPipe hand detection
import mediapipe as mp
mp_hands = mp.solutions.hands

DIFF_THRESH = 20
MOTION_RATIO_THRESH = 0.01
MIN_STATIC_LEN = 6

# Instruction templates
content_grab_and_move = [
    "Instruction: pick this up and put it there.",
    "Instruction: take this one and place it on that.",
    "Instruction: pick it up and put it over there.",
    "Instruction: carry this and release it there.",
    "Instruction: stack this block over there.",
    "Instruction: move this onto that one.",
]

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

def detect_activity_start_from_image_changes(frames):
    """
    Detect activity start based on image changes.
    Returns: frame index for activity start based on image changes.
    """
    if len(frames) < 5:
        return 0
    
    # Compute per-frame differences from the previous frame.
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
    
    # Insert a placeholder for the first element.
    diffs.insert(0, 0)
    
    # Find the activity start frame (consecutive frames above threshold).
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
    
    # If no clear activity start is found, use a default value.
    if activity_start_from_image is None:
        activity_start_from_image = min(5, len(frames) - 1)
    
    return activity_start_from_image

def process_video_frames(frames, frame_num=2):
    """
    Main entry for processing a list of video frames (multi-pause-frame version).

    Args:
        frames: list of np.array - BGR frames as NumPy arrays.
        frame_num: int - number of pause frames to extract (1, 2, or 3), default 2.

    Returns:
        tuple: (pause_frames, keypoint_vectors, thought)
            pause_frames: list of np.array - raw images for each pause frame.
            keypoint_vectors: list of np.array - 12-D keypoint vectors for each pause frame.
            thought: str - randomly chosen instruction text.

    Note: returns (None, None, None) on failure.
    """
    if not frames or len(frames) == 0:
        print("错误: 输入帧列表为空")
        return None, None, None
    
    if frame_num not in [1, 2, 3]:
        print(f"错误: 不支持的停顿帧数量 {frame_num}，仅支持 1, 2, 3")
        return None, None, None
    
    n_frames = len(frames)
    
    # Step 1: detect hand keypoints and determine the hand interval.
    all_frames, all_keypoints, valid_mask, hand_start, hand_end = process_video_frames_with_mediapipe(frames)
    
    # If hand detection fails, try image-change detection.
    if hand_start is None or hand_end is None:
        print("警告: 未检测到手部，尝试使用图像变化检测")
        hand_start = detect_activity_start_from_image_changes(frames)
        hand_end = n_frames - 1
    else:
        # Combine both methods to determine hand start.
        hand_start1 = detect_activity_start_from_image_changes(frames)
        hand_start = min(hand_start, hand_start1)
    
    print(f"手部检测区间: 第 {hand_start} 帧到第 {hand_end} 帧")
    
    if hand_start is None or hand_end is None:
        print("错误: 无法确定手部区间")
        return None, None, None
    
    # Step 2: interpolate keypoints.
    keypoints_interp = interpolate_keypoints(all_keypoints, valid_mask)
    
    # Step 3: detect pause frames within the hand interval (using frame_num).
    pause_frame_indices = extract_pause_frames_in_interval(all_frames, hand_start, hand_end, frame_num)
    
    if len(pause_frame_indices) < frame_num:
        print(f"警告: 未检测到足够的停顿帧（当前 {len(pause_frame_indices)}，需要 {frame_num}），使用补充策略")
        # Fill in missing frames.
        while len(pause_frame_indices) < frame_num:
            if len(pause_frame_indices) == 0:
                pause_frame_indices.append(hand_start + 5)
            else:
                # Add after the last frame.
                next_frame = min(pause_frame_indices[-1] + 20, hand_end - 5)
                if next_frame not in pause_frame_indices and next_frame > pause_frame_indices[-1]:
                    pause_frame_indices.append(next_frame)
                else:
                    pause_frame_indices.append(hand_end)
                    break
    
    print(f"选定的停顿帧索引: {pause_frame_indices}")
    
    # Step 4: randomly select a thought instruction.
    thought = np.random.choice(content_grab_and_move)
    print(f"选择的指令: {thought}")
    
    # Step 5: extract pause frames and keypoint vectors.
    pause_frames = []
    keypoint_vectors = []
    
    for frame_idx in pause_frame_indices[:frame_num]:  # Ensure only the requested count.
        if frame_idx >= len(frames):
            print(f"警告: 帧索引 {frame_idx} 超出范围，使用最后一帧")
            frame_idx = len(frames) - 1
        
        # Get pause frame (raw image).
        pause_frame = frames[frame_idx].copy()
        
        # Get corresponding keypoint vector.
        if frame_idx < len(keypoints_interp):
            kp_vector = keypoints_interp[frame_idx].copy()
        else:
            kp_vector = np.zeros(12, dtype=float)
            print(f"警告: 帧索引 {frame_idx} 超出关键点数组范围，使用零向量")
        
        keypoint_vectors.append(kp_vector)

        # Draw keypoints on the pause frame.
        for k in range(0, 12, 3):
            # Convert normalized coordinates to pixel coordinates.
            x = int(kp_vector[k] * pause_frame.shape[1])
            y = int(kp_vector[k+1] * pause_frame.shape[0])
            
            # Clamp coordinates to image bounds.
            x = max(0, min(x, pause_frame.shape[1] - 1))
            y = max(0, min(y, pause_frame.shape[0] - 1))
            
            # Draw green dot (radius 4, filled).
            cv2.circle(pause_frame, (x, y), 4, (0, 255, 0), -1)
        
        pause_frames.append(pause_frame)
    
    # Ensure the specified number of elements is returned.
    while len(pause_frames) < frame_num:
        pause_frames.append(frames[-1].copy())
        keypoint_vectors.append(np.zeros(12, dtype=float))
    
    return pause_frames, keypoint_vectors, thought

def process_video_frames_with_mediapipe(frames):
    """
    Process frames and detect hand keypoints (with optional masking).

    Args:
        frames: list of np.array - video frames.

    Returns:
        tuple: (all_frames, all_keypoints, valid_mask, hand_start, hand_end)
    """
    n_frames = len(frames)
    
    # Initialize MediaPipe hand detector.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        model_complexity=1
    )
    
    all_frames = []
    all_keypoints = np.zeros((n_frames, 12), dtype=float)
    valid_mask = np.zeros(n_frames, dtype=bool)
    hand_detected_frames = []
    
    print(f"正在处理 {n_frames} 帧")
    
    for i in tqdm(range(n_frames), desc="帧处理"):
        frame = frames[i]

        # Keep consistent with pauseframes.py: use mask by default.
        masked_frame = apply_mask_to_frame(frame)
        all_frames.append(masked_frame)

        img_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        # Detect hand keypoints.
        results = hands.process(img_rgb)
        
        hand_detected = results.multi_hand_landmarks is not None
        if hand_detected:
            hand_detected_frames.append(i)
            valid_mask[i] = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract keypoints: index fingertip (8), first joint (7), second joint (6), wrist (0).
            mapping = [
                (8, 0, 1, 2),   # Index fingertip -> indices 0,1,2 (x,y,z)
                (7, 3, 4, 5),   # Index first joint -> indices 3,4,5
                (6, 6, 7, 8),   # Index second joint -> indices 6,7,8
                (0, 9, 10, 11)  # Wrist -> indices 9,10,11
            ]
            
            for lm_idx, x_idx, y_idx, z_idx in mapping:
                lm = hand_landmarks.landmark[lm_idx]
                all_keypoints[i, x_idx] = lm.x
                all_keypoints[i, y_idx] = lm.y
                all_keypoints[i, z_idx] = lm.z
    
    hands.close()
    
    # Determine the hand interval.
    if not hand_detected_frames:
        print("警告: 未检测到手部")
        return all_frames, all_keypoints, valid_mask, None, None
    
    hand_start = max(0, hand_detected_frames[0])  
    hand_end = hand_detected_frames[-1]
    
    # Ensure the interval has enough frames.
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
        
        # Keep earlier frames at 0 (no interpolation).
        keypoints_interp[:first_valid] = 0.0
        
        # Interpolate from first_valid to the end.
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
    Detect pause frames within the hand interval using frame-difference signals.

    Args:
        all_frames: all video frames
        hand_start: start frame index of hand interval
        hand_end: end frame index of hand interval
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
         if none, use hand_start+5 and hand_end-10
    - 3 frames: if >=3 segments, use centers of first, second, and last;
         if 2 segments, use first center, second center, hand_end-10;
         if <=1, distribute evenly between hand_start and hand_end
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
                return [first_mid, max(first_mid + 10, hand_end - 10)]
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
            # Two segments: first center, second center, hand_end-10.
            first_mid = get_segment_center(static_segments[0])
            second_mid = get_segment_center(static_segments[-1])
            last = max(second_mid + 10, hand_end - 10)
            return [first_mid, second_mid, last]
        else:
            # No static segments; distribute evenly (matches pauseframes.py).
            total_len = hand_end - hand_start
            return [
                hand_start + 5,
                hand_start + total_len // 2,
                hand_end - 10
            ]
    
    else:
        raise ValueError(f"不支持的停顿帧数量: {num_pause_frames}，仅支持 1, 2, 3")

# Example usage function
def example_usage(video_path, frame_num=2):
    """
    Example: how to read a video and call the processing function.

    Args:
        video_path: video file path
        frame_num: number of pause frames to extract (1, 2, or 3)
    """
    # Read video file.
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    print(f"读取了 {len(frames)} 帧")
    
    # Call the processing function.
    pause_frames, keypoint_vectors, thought = process_video_frames(frames, frame_num)
    
    if pause_frames is not None:
        print(f"返回了 {len(pause_frames)} 个停顿帧")
        print(f"每个关键点向量维度: {keypoint_vectors[0].shape}")
        print(f"Thought: {thought}")
        
        # Further processing can be done here.
        return pause_frames, keypoint_vectors, thought
    else:
        print("处理失败")
        return None, None, None

# Main entry when run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从视频中提取多个停顿帧')
    parser.add_argument('--video_path', type=str, required=True,
                        help='视频文件路径')
    parser.add_argument('--frame_num', type=int, default=2, choices=[1, 2, 3],
                        help='需要提取的停顿帧数量 (1, 2, 或 3)')
    
    args = parser.parse_args()
    
    result = example_usage(args.video_path, args.frame_num)
