
"""
    Segment objects by clicking on the image.
    Store segmented area pixel coordinates in .npy file.
"""



import os
import cv2
import torch
import numpy as np
import json
import sys
from urllib.request import urlretrieve



# --- Helper Function to Download Model ---
def download_checkpoint(ckpt_path, url):
    """Downloads the model checkpoint if it doesn't exist."""
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Downloading...")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        try:
            urlretrieve(url, ckpt_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading checkpoint: {e}")
            print("Please download it manually from the SAM 2 repository and place it in the 'checkpoints' folder.")
            sys.exit(1)
    else:
        print(f"Checkpoint already exists at {ckpt_path}. Skipping download.")

# --- Main Segmentation Function ---
def segment_with_point_prompt(image_path, checkpoint_path, model_config_path, output_dir):
    """
    Loads an image, allows the user to click a point, and segments the region
    using SAM 2, saving all pixel coordinates within the mask to a specified directory.
    """
    # --- 1. Setup Model ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError:
        print("SAM 2 library not found.")
        print("Please install it by running: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        sys.exit(1)

    # Build and load the model.
    predictor = SAM2ImagePredictor(build_sam2(model_config_path, checkpoint_path))
    print("SAM 2 model loaded successfully.")

    # --- 2. Load Image ---
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    display_image = image_bgr.copy()

    # --- 3. Interactive Point Selection ---
    clicked_point = None
    all_pixel_coordinates = [] # Will store all pixels in the mask

    def mouse_callback(event, x, y, flags, param):
        """Handles mouse clicks to trigger segmentation."""
        nonlocal clicked_point, display_image, all_pixel_coordinates
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Reset the display image on each click
            display_image = image_bgr.copy()
            
            clicked_point = [x, y]
            print(f"Clicked point: {clicked_point}")

            # --- 4. Run Segmentation ---
            predictor.set_image(image_rgb)
            
            input_points = torch.tensor([[[x, y]]], device=DEVICE)
            input_labels = torch.tensor([[1]], device=DEVICE)

            print("Running model prediction...")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=False,
                )
            
            mask_cpu = masks[0].astype(np.uint8)
            print(f"Prediction complete. Best score: {scores[0]:.3f}")

            # --- 5. Extract and Visualize Coordinates ---
            # Find all pixel coordinates within the mask
            rows, cols = np.where(mask_cpu > 0)
            # We store all pixels, converting numpy's (row, col) to image's (x, y) format
            all_pixel_coordinates = list(zip(cols.tolist(), rows.tolist()))

            # Find contours (outlines) for visualization purposes only
            contours, _ = cv2.findContours(mask_cpu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Draw the contour on the display image for visual feedback
                cv2.drawContours(display_image, [largest_contour], -1, (0, 255, 0), 2)
                cv2.circle(display_image, tuple(clicked_point), 5, (0, 0, 255), -1)
                print(f"Found region with {len(all_pixel_coordinates)} pixels. Press 's' to save or click again.")
            else:
                print("No contour found for the generated mask.")
                all_pixel_coordinates = [] # Clear if no region found
            
            # Refresh the image window with the new drawings
            cv2.imshow("Image", display_image)


    # --- 6. Display and Interaction Loop ---
    cv2.imshow("Image", display_image)
    cv2.setMouseCallback("Image", mouse_callback)
    
    print("\nClick on a region in the image to segment it.")
    print("Press 's' to save the coordinates of the last selected region.")
    print("Press 'q' to quit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            if all_pixel_coordinates:
                # *** UPDATED SAVING LOGIC ***
                # Create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                
                # Construct the full path for the output file
                output_path = os.path.join(output_dir, "segmented_pixel_coords.npy")
                
                # Convert the list of coordinates to a NumPy array
                coords_array = np.array(all_pixel_coordinates, dtype=np.int32)
                
                # Save the array using np.save
                np.save(output_path, coords_array)
                print(f"Successfully saved {len(all_pixel_coordinates)} pixel coordinates to {output_path}")
            else:
                print("No region has been selected yet. Click on the image first.")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- Configuration ---
    # This robustly finds the project root by going up one level from the script's directory ('utils')
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    IMAGE_TO_SEGMENT = os.path.join(PROJECT_ROOT, 'data', 'frame_0000.png')
    
    # The SAM2 repo is now in 'sam2_repo' to avoid import conflicts.
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'sam2_repo', 'checkpoints')
    
    # *** NEW: Define the output directory for segmentation data ***
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'seg_data')
    
    # SAM 2 Hiera Large model details
    CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
    MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/segment_anything_2/assets/models/{CHECKPOINT_NAME}"
    
    checkpoint_full_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    download_checkpoint(checkpoint_full_path, CHECKPOINT_URL)

    # Pass the new output directory to the main function
    segment_with_point_prompt(IMAGE_TO_SEGMENT, checkpoint_full_path, MODEL_CONFIG, OUTPUT_DIR)
