import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict, annotate

def apply_mask_to_frame(frame):
    """
    Apply a mask to the frame and zero out the masked region
    to avoid detecting the operator's hand.
    """


    mask_polygon = np.array([
        [0, 480],
        [1280, 480],
        [1280, 720],
        [0, 720]
    ], dtype=np.int32)
    
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    cv2.fillPoly(mask, [mask_polygon], 0)


    # Apply mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)


    return masked_frame

def main():
    """Run a lightweight GroundingDINO prompt sweep for inspection."""
    # Configuration
    MODEL_CONFIG_PATH = "hand_VLA/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    MODEL_CHECKPOINT_PATH = "hand_VLA/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    DATA_ROOT = Path("hand_VLA/first_frame_0214_jelly") 
    OUTPUT_ROOT = Path("hand_VLA/data_generator/testDINO") 
    
    # Slightly lower thresholds to improve recall
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3
    
    # Prompts to test
    PROMPTS_TO_TEST = [
        "cup of jelly"
    ]
    
    # Create output directory
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading GroundingDINO model...")
    model = load_model(MODEL_CONFIG_PATH, MODEL_CHECKPOINT_PATH)
    
    # Collect episode directories
    episode_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    print(f"Found {len(episode_dirs)} episode directories")
    
    # Process each episode
    max_episodes = 20
    for episode_idx, episode_dir in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
        if episode_idx >= max_episodes: 
            break
            
        # Resolve RGB image path
        rgb_path = episode_dir / "right_rgb_frame_0.png"
        if not rgb_path.exists():
            rgb_path = episode_dir / "rgb_frame_0.png"
            if not rgb_path.exists():
                print(f"Warning: {rgb_path} does not exist, skipping episode")
                continue
        
        # Read original image
        original_image = cv2.imread(str(rgb_path))
        if original_image is None:
            print(f"Warning: cannot read image {rgb_path}, skipping episode")
            continue
        
        # Apply mask and save temporary file
        masked_image = apply_mask_to_frame(original_image)
        temp_masked_path = episode_dir / "temp_masked_image.png"
        cv2.imwrite(str(temp_masked_path), masked_image)
        
        # Load original image (for drawing)
        image_source, _ = load_image(str(rgb_path))
        
        # Load masked image (for detection)
        _, image = load_image(str(temp_masked_path))
        
        # Test each prompt
        for prompt_idx, prompt in enumerate(PROMPTS_TO_TEST):
            # Create output directory for current prompt
            safe_prompt_name = prompt.replace(" . ", "_").replace(" ", "_")
            prompt_output_dir = OUTPUT_ROOT / f"{prompt_idx:02d}_{safe_prompt_name}" 
            prompt_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Object detection
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            
            # Draw bounding boxes on original image
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            
            # Save annotated image
            output_path = prompt_output_dir / f"{episode_dir.name}.jpg"
            cv2.imwrite(str(output_path), annotated_frame)
            
        # Remove temporary file
        temp_masked_path.unlink()
            
    print("All processing completed. Results saved to:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()
