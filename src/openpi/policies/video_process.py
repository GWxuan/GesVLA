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

def process_video_frames(frames):
    """
    Main entry for processing a list of video frames.

    Args:
        frames: list of np.array - BGR frames as NumPy arrays.

    Returns:
        tuple: (pause_frames, keypoint_vectors, thought)
            pause_frames: list of np.array - two pause-frame images.
            keypoint_vectors: list of np.array - 12-D keypoint vectors for each pause frame.
            thought: str - randomly chosen instruction text.

    Note: returns (None, None, None) on failure.
    """
    if not frames or len(frames) == 0:
        print("错误: 输入帧列表为空")
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
    
    # Step 3: detect pause frames within the hand interval.
    pause_frame_indices = extract_pause_frames_in_interval(frames, hand_start, hand_end)
    
    if len(pause_frame_indices) < 2:
        print(f"警告: 未检测到足够的停顿帧，使用手部区间边界")
        pause_frame_indices = [hand_start, hand_end]
    elif len(pause_frame_indices) > 2:
        # If multiple pause frames are detected, take the first and last.
        pause_frame_indices = [pause_frame_indices[0], pause_frame_indices[-1]]
    
    print(f"选定的停顿帧索引: {pause_frame_indices}")
    
    # Step 4: randomly select a thought instruction.
    thought = np.random.choice(content_grab_and_move)
    print(f"选择的指令: {thought}")
    
    # Step 5: extract pause frames and keypoint vectors.
    pause_frames = []
    keypoint_vectors = []
    
    for frame_idx in pause_frame_indices[:2]:  # Only take the first two.
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
        
    
    # Ensure two elements are returned.
    while len(pause_frames) < 2:
        pause_frames.append(frames[-1].copy())
        keypoint_vectors.append(np.zeros(12, dtype=float))
    
    return pause_frames, keypoint_vectors, thought

def process_video_frames_with_mediapipe(frames):
    """Process frames and detect hand keypoints."""
    n_frames = len(frames)
    
    # Initialize MediaPipe hand detector.
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
        model_complexity=1
    )
    
    all_keypoints = np.zeros((n_frames, 12), dtype=float)
    valid_mask = np.zeros(n_frames, dtype=bool)
    hand_detected_frames = []
    
    print(f"正在处理 {n_frames} 帧")
    
    for i in tqdm(range(n_frames), desc="帧处理"):
        frame = frames[i]
        
        # Detect hand keypoints.
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        return frames, all_keypoints, valid_mask, None, None
    
    hand_start = max(20, hand_detected_frames[0])  
    hand_end = hand_detected_frames[-1]
    
    # Ensure the interval has enough frames.
    if hand_end - hand_start < MIN_STATIC_LEN * 2:
        print(f"警告: 手部区间太短 ({hand_end - hand_start} 帧)")
        return frames, all_keypoints, valid_mask, hand_start, hand_end
    
    return frames, all_keypoints, valid_mask, hand_start, hand_end

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

def extract_pause_frames_in_interval(all_frames, hand_start, hand_end):
    """
    Detect pause frames within the hand interval using frame differences.
    Returns: list of pause frame indices.
    """
    if hand_start is None or hand_end is None:
        return []
    
    # Extract frames within the hand interval.
    interval_frames = all_frames[hand_start:hand_end+1]
    
    if len(interval_frames) < MIN_STATIC_LEN * 2:
        print(f"警告: 手部区间帧数不足 ({len(interval_frames)} 帧)")
        return []
    
    static_segments = []
    current_static_start = None
    
    # Convert the first frame to grayscale.
    prev_gray = cv2.cvtColor(interval_frames[0], cv2.COLOR_BGR2GRAY)
    
    # Iterate over frames in the hand interval.
    for i in range(1, len(interval_frames)):
        frame = interval_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute inter-frame differences.
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_bin = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(diff_bin)
        motion_ratio = non_zero / diff_bin.size
        
        # If the difference is small, treat as static.
        if motion_ratio < MOTION_RATIO_THRESH:
            if current_static_start is None:
                current_static_start = i
        else:
            if current_static_start is not None:
                seg_len = i - current_static_start
                if seg_len >= MIN_STATIC_LEN:
                    # Convert to original video frame indices.
                    static_segments.append((
                        current_static_start + hand_start, 
                        i - 1 + hand_start
                    ))
                current_static_start = None
        
        prev_gray = gray
    
    # If the video ends while still static, record the segment.
    if current_static_start is not None:
        seg_len = len(interval_frames) - current_static_start
        if seg_len >= MIN_STATIC_LEN:
            static_segments.append((
                current_static_start + hand_start, 
                hand_end
            ))
    
    print(f"检测到静止段: {static_segments}")
    
    # Select pause frames.
    pause_frames = []
    if len(static_segments) >= 2:
        # Use mid-frames of the first and last static segments.
        first_mid = (static_segments[0][0] + static_segments[0][1]) // 2
        last_mid = (static_segments[-1][0] + static_segments[-1][1]) // 2
        if last_mid - first_mid >= 5:
            pause_frames = [first_mid, last_mid]
        else:
            pause_frames = [first_mid, hand_end]
    elif len(static_segments) == 1:
        mid = (static_segments[0][0] + static_segments[0][1]) // 2
        pause_frames = [mid, hand_end]
    else:
        # If no static segment is detected.
        mid = hand_start + 5
        pause_frames = [mid, hand_end-10]
    
    # Ensure pause frames lie within the hand interval.
    pause_frames = [max(hand_start, min(p, hand_end)) for p in pause_frames]
    
    return pause_frames

# Example usage function
def example_usage(video_path):
    """Example: read a video and call the processing function."""
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
    pause_frames, keypoint_vectors, thought = process_video_frames(frames)
    
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
    # Example call — replace with your video path.
    video_path = "data/datasets/example/episode_10/gesture/video.mp4"
    result = example_usage(video_path)