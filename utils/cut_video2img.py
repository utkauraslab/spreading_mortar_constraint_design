# import cv2
# import os
# import sys






# def cut_video_into_frames(video_path, output_folder):
#     """
#     Cuts a video file into individual frames and saves them as images.

#     Args:
#         video_path (str): /home/farong/Desktoop/spreading_mortar_constraint_design
#         output_folder (str): /home/farong/Desktoop/spreading_mortar_constraint_design/data
#     """
#     # --- 1. Validate Input and Create Output Directory ---
#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found at '{video_path}'")
#         return

#     if not os.path.exists(output_folder):
#         print(f"Output folder '{output_folder}' not found. Creating it...")
#         os.makedirs(output_folder)
#         print(f"Successfully created directory: '{output_folder}'")

#     # --- 2. Open the Video File ---
#     video_capture = cv2.VideoCapture(video_path)

#     # Check if the video was opened successfully
#     if not video_capture.isOpened():
#         print(f"Error: Could not open video file '{video_path}'")
#         return

#     # --- 3. Get Video Properties (for progress tracking) ---
#     total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = video_capture.get(cv2.CAP_PROP_FPS)
#     print(f"Video Info: {total_frames} total frames at {fps:.2f} FPS.")


#     # --- 4. Read and Save Frames ---
#     frame_count = 0
#     success = True
#     while success:
#         # Read the next frame
#         success, frame = video_capture.read()

#         if success:
#             # Construct the output filename
#             # Using zfill to pad with leading zeros (e.g., 00001, 00002) for proper sorting
#             #frame_filename = os.path.join(output_folder, f"frame_{str(frame_count).zfill(5)}.jpg")

#             frame_filename = os.path.join(output_folder, f"frame_{str(frame_count).zfill(4)}.png")

#             # Save the frame as a JPEG image
#             cv2.imwrite(frame_filename, frame)

#             frame_count += 1

#             # Print progress
#             if frame_count % 100 == 0:
#                 print(f"Processed {frame_count}/{total_frames} frames...")

#     # --- 5. Release Resources ---
#     video_capture.release()
#     print(f"\nDone! Extracted {frame_count} frames and saved them to '{output_folder}'.")


# # --- Example Usage ---
# if __name__ == '__main__':
    
#     # __file__ : /home/farong/Desktoop/spreading_mortar_constraint_design/utils/cut_video2img.py
#     # os.path.dirname(__file__): /home/farong/Desktoop/spreading_mortar_constraint_design/utils
#     # os.path.dirname(os.path.dirname(__file__)): /home/farong/Desktoop/spreading_mortar_constraint_design
#     project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


#     # --- Configuration ---
#     # IMPORTANT: Replace this with the actual name of your video file
#     video_filename = "spreading_mortar_videos/mortar2.mp4"
#     # The video is expected to be in a 'videos' subdirectory
#     input_video_path = os.path.join(project_directory, video_filename)

#     # Define the directory to save the frames
#     output_frames_directory = os.path.join(project_directory, 'data')

#     # --- Run the function ---
#     # Check if a video file name is provided as a command-line argument
#     if len(sys.argv) > 1:
#         input_video_path = sys.argv[1]
#         print(f"Using video file from command-line argument: {input_video_path}")
#     elif not os.path.exists(input_video_path):
#          print("---")
#          print(f"ERROR: The example video '{video_filename}' was not found at '{input_video_path}'.")
#          print("Please do one of the following:")
#          print(f"1. Place your video file at that location and rename it to '{video_filename}'.")
#          print("2. Update the 'video_filename' variable in this script with your video's name.")
#          print("3. Run the script with the path to your video as an argument, e.g.:")
#          print(f"   python {os.path.basename(__file__)} /path/to/your/video.mp4")
#          print("---")
#          sys.exit(1) # Exit the script if the default video isn't found

#     cut_video_into_frames(video_path=input_video_path, output_folder=output_frames_directory)












import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import sys

# --- Path Setup ---
# Establish the project root directory by going up one level from the current script's location.
# __file__ is the path to the current script: /.../spreading_mortar_constraint_design/utils/get_depth_map.py
# os.path.dirname(__file__) is the directory of the script: /.../spreading_mortar_constraint_design/utils
# The '..' goes up one level to the project root: /.../spreading_mortar_constraint_design
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the Depth-Anything-V2 submodule directory to the Python path to allow for direct imports
# This is more specific and safer than adding the whole project root.
DEPTH_ANYTHING_PATH = os.path.join(PROJECT_ROOT, 'Depth-Anything-V2')
sys.path.append(DEPTH_ANYTHING_PATH)

try:
    # Now the import should work because the submodule's directory is on the path
    from depth_anything_v2.dpt import DepthAnythingV2
except ModuleNotFoundError:
    print("Error: 'depth_anything_v2' module not found. Please ensure you have set up the submodules correctly:")
    print(f"1. The submodule should exist at: {DEPTH_ANYTHING_PATH}")
    print("2. You may need to install its dependencies: cd {DEPTH_ANYTHING_PATH} && pip install -e .")
    sys.exit(1)


# --- Load Data ---
# Load tensors and models using paths relative to the PROJECT_ROOT
vertex_coords_path = os.path.join(PROJECT_ROOT, "trowel_tip_keypoints_traj_3d.pt") # Assuming this is the correct file
vertex_coords_tensor = torch.load(vertex_coords_path)
num_keypoints, num_frames, _ = vertex_coords_tensor.shape
print(f"Number of keypoints: {num_keypoints}, Number of frames: {num_frames}")

# --- Load DepthAnythingV2 Model ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
encoder = 'vitl' # Using the 'Large' model
model = DepthAnythingV2(**model_configs[encoder])

# Construct the full path to the checkpoint file
ckpt_path = os.path.join(PROJECT_ROOT, 'Depth-Anything-V2', 'checkpoints', f'depth_anything_v2_{encoder}.pth')
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please run 'setup_dependencies.sh' to download it.")
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model = model.to(DEVICE).eval()

# --- Prepare Output ---
depth_tensor = torch.zeros((num_keypoints, num_frames), dtype=torch.float32)
depth_maps_all = []

# Define input and output folders relative to the PROJECT_ROOT
image_folder = os.path.join(PROJECT_ROOT, "visualized_frames")
depth_output_folder = os.path.join(PROJECT_ROOT, "depth_images")
os.makedirs(depth_output_folder, exist_ok=True)


# --- Main Processing Loop ---
for f in tqdm(range(num_frames), desc="Extracting depths"):
    # Corrected frame filename format to use 5 digits (e.g., frame_00000.png)
    frame_path = os.path.join(image_folder, f"frame_{f:05d}.png")
    if not os.path.exists(frame_path):
        print(f"Warning: Missing frame {frame_path}, skipping")
        continue
    
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Warning: Could not load {frame_path}, skipping")
        continue

    # Run depth prediction
    depth_raw = model.infer_image(img)  # Shape: [H, W] numpy array

    # Invert to standard depth (higher values are farther away)
    depth_relative = np.max(depth_raw) - depth_raw

    # --- Metric Depth Calculation ---
    # A known point on the image (px, py) corresponds to a known real-world depth
    known_depth_meters = 0.15 # Example: 15cm
    px, py = (979, 632) # Coordinates of the known point

    relative_value_at_known_point = depth_relative[py, px]

    # Avoid division by zero
    if relative_value_at_known_point > 1e-6:
        scale_factor = known_depth_meters / relative_value_at_known_point
        depth_metric = depth_relative * scale_factor
    else:
        depth_metric = depth_relative # Fallback if the known point has no depth

    # --- Visualization and Data Extraction ---
    # Normalize for visualization
    depth_min, depth_max = np.min(depth_metric), np.max(depth_metric)
    if depth_max > depth_min:
        depth_map_normalized = ((depth_metric - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    else:
        depth_map_normalized = np.zeros_like(depth_metric, dtype=np.uint8)

    depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

    H, W, _ = img.shape
    for k in range(num_keypoints):
        # Using integer conversion for pixel coordinates
        x, y = int(vertex_coords_tensor[k, f, 0]), int(vertex_coords_tensor[k, f, 1])
        if 0 <= x < W and 0 <= y < H:
            depth_tensor[k, f] = float(depth_metric[y, x])
        else:
            depth_tensor[k, f] = float('nan')

        # Visualize depth point on the color map
        cv2.circle(depth_map_colored, (x, y), 4, (0, 0, 255), -1)
        
    # Save colored depth map
    depth_output_path = os.path.join(depth_output_folder, f"depth_{f:05d}_colored.png")
    cv2.imwrite(depth_output_path, depth_map_colored)
    
    depth_maps_all.append(torch.from_numpy(depth_metric).unsqueeze(0))

# --- Save Final Tensors ---
# Save results to the PROJECT_ROOT
depth_maps_tensor = torch.cat(depth_maps_all, dim=0)
torch.save(depth_maps_tensor, os.path.join(PROJECT_ROOT, "depth_map_cross_frames.pt"))
print(f"Saved per-frame depth maps tensor: {depth_maps_tensor.shape}")

torch.save(depth_tensor, os.path.join(PROJECT_ROOT, "trowel_tip_keypoints_depth_tensor.pt"))
print(f"Saved keypoint depth tensor: {depth_tensor.shape}")