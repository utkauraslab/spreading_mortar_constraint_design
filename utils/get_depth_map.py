




# import os
# import torch
# import cv2
# import numpy as np
# from tqdm import tqdm
# import sys


# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add the Depth-Anything-V2 submodule directory to the Python path to allow for direct imports
# # This is more specific and safer than adding the whole project root.
# DEPTH_ANYTHING_PATH = os.path.join(PROJECT_ROOT, 'Depth-Anything-V2')
# sys.path.append(DEPTH_ANYTHING_PATH)

# try:
#     # Now the import should work because the submodule's directory is on the path
#     from depth_anything_v2.dpt import DepthAnythingV2
# except ModuleNotFoundError:
#     print("Error: 'depth_anything_v2' module not found. Please ensure you have set up the submodules correctly:")
#     print(f"1. The submodule should exist at: {DEPTH_ANYTHING_PATH}")
#     print("2. You may need to install its dependencies: cd {DEPTH_ANYTHING_PATH} && pip install -e .")
#     sys.exit(1)

# # Load coords 2D coordinate tensor 
# vertex_coords_tensor = torch.load("keypoints_2d_traj.pt")
# num_keypoints, num_frames, _ = vertex_coords_tensor.shape
# print(f"Number of keypoints: {num_keypoints}, Number of frames: {num_frames}")

# # Load DepthAnythingV2 Model 
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {DEVICE}")

# model_configs = {
#     'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
#     'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
#     'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
#     'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
# }
# encoder = 'vitl'
# model = DepthAnythingV2(**model_configs[encoder])
# ckpt_path = os.path.join('Depth-Anything-V2/checkpoints', f'depth_anything_v2_{encoder}.pth')
# if not os.path.exists(ckpt_path):
#     raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Download it from the official repository.")
# model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
# model = model.to(DEVICE).eval()

# # output tensor
# depth_tensor = torch.zeros((num_keypoints, num_frames), dtype=torch.float32)
# depth_maps_all = []

# # output paths 
# image_folder = "./data"
# depth_output_folder = "./depth_images"
# os.makedirs(depth_output_folder, exist_ok=True)


# for f in tqdm(range(num_frames), desc="Extracting depths"):
#     frame_path = os.path.join(image_folder, f"frame_{f:04d}.png")
#     if not os.path.exists(frame_path):
#         print(f"Warning: Missing frame {frame_path}, skipping")
#         continue
    
#     img = cv2.imread(frame_path)
#     if img is None:
#         print(f"Warning: Could not load {frame_path}, skipping")
#         continue

#     # Run depth prediction
#     depth_raw = model.infer_image(img)  # Shape: [H, W] numpy array in meters

#     # print(np.max(depth_map))
#     # break

#     # Correct inverted depth map to standard depth
#     #max_depth_raw = np.max(depth_map)
#     #depth_relative = max_depth_raw - depth_map  # Invert to standard depth
#     #depth_map_corrected *= 1000  # Convert from meters to mm

#     depth_relative = np.max(depth_raw) - depth_raw

#     known_depth_meters = 0.15

#     px, py = (979, 632)

#     relative_value_at_known_point = depth_relative[py, px]

#     scale_factor = known_depth_meters / relative_value_at_known_point

#     depth_metric = depth_relative * scale_factor

#     # Normalize depth map for color mapping (0 to 255 range)
#     depth_min, depth_max = np.min(depth_metric), np.max(depth_metric)
#     if depth_max > depth_min:  # Avoid division by zero
#         depth_map_normalized = ((depth_metric - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
#     else:
#         depth_map_normalized = np.zeros_like(depth_metric, dtype=np.uint8)

#     # Apply color map (e.g., JET colormap)
#     depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

#     H, W, _ = img.shape
#     for k in range(num_keypoints):
#         x, y = vertex_coords_tensor[k, f].tolist()
#         if 0 <= x < W and 0 <= y < H:
#             depth_tensor[k, f] = float(depth_metric[y, x])
#         else:
#             depth_tensor[k, f] = float('nan')  # Use NaN for out-of-bounds

#         # Visualize depth point on the color map
#         cv2.circle(depth_map_colored, (x, y), 4, (0, 0, 255), -1)  # Red dot
        
    
#     # Save colored depth map
#     depth_output_path = os.path.join(depth_output_folder, f"depth_{f:04d}_colored.png")
#     cv2.imwrite(depth_output_path, depth_map_colored)
    
    
#     # Save depth map tensor
#     depth_maps_all.append(torch.from_numpy(depth_metric).unsqueeze(0)) # 1, h, w


# depth_maps_tensor = torch.cat(depth_maps_all, dim=0)
# torch.save(depth_maps_tensor, "depth_map_cross_frames.pt")
# print(f"Saved per-frame depth maps tensor: {depth_maps_tensor.shape}") # 70. h, w

    
# torch.save(depth_tensor, "keypoints_depth_tensor.pt")
# print(f"Saved keypoint depth tensor: {depth_tensor.shape}")














import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
import sys

# --- Path Setup ---
# Establish the project root directory by going up one level from the current script's location.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the Depth-Anything-V2 submodule directory to the Python path
DEPTH_ANYTHING_PATH = os.path.join(PROJECT_ROOT, 'Depth-Anything-V2')
sys.path.append(DEPTH_ANYTHING_PATH)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ModuleNotFoundError:
    print("Error: 'depth_anything_v2' module not found. Please ensure you have set up the submodules correctly.")
    sys.exit(1)


# --- Load Data ---
# *** UPDATED TO LOAD .NPY FILE ***
# Define the path to the .npy file from the CoTracker script
vertex_coords_path = os.path.join(PROJECT_ROOT, "keypoints_2d_traj.npy") 
if not os.path.exists(vertex_coords_path):
    raise FileNotFoundError(f"Keypoints file not found at {vertex_coords_path}")

# Load the numpy array
vertex_coords_array = np.load(vertex_coords_path)
# Convert to a PyTorch tensor for compatibility with the rest of the script
vertex_coords_tensor = torch.from_numpy(vertex_coords_array)

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
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model = model.to(DEVICE).eval()

# --- Prepare Output ---
depth_tensor = torch.zeros((num_keypoints, num_frames), dtype=torch.float32)
depth_maps_all = []

# Define input and output folders relative to the PROJECT_ROOT
image_folder = os.path.join(PROJECT_ROOT, "data")
depth_output_folder = os.path.join(PROJECT_ROOT, "depth_images")
os.makedirs(depth_output_folder, exist_ok=True)


# --- Main Processing Loop ---
for f in tqdm(range(num_frames), desc="Extracting depths"):
    # *** CORRECTED LINE ***
    # (e.g., frame_0000.png)
    frame_path = os.path.join(image_folder, f"frame_{f:04d}.png")
    if not os.path.exists(frame_path):
        # This warning will print for each missing frame
        if f == 0: # Print a more detailed message for the first missing frame
            print(f"\nWarning: Missing frame {frame_path}. The script will skip all missing frames.")
            print("Please ensure your frame images are in the 'visualized_frames' directory and are named correctly (e.g., frame_00000.png).")
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
    depth_output_path = os.path.join(depth_output_folder, f"depth_{f:04d}_colored.png")
    cv2.imwrite(depth_output_path, depth_map_colored)
    
    depth_maps_all.append(torch.from_numpy(depth_metric).unsqueeze(0))

# --- Add a check before saving ---
if not depth_maps_all:
    raise RuntimeError(
        "No depth maps were generated. The 'depth_maps_all' list is empty.\n"
        "This is likely because the script could not find any of the input frame images.\n"
        f"Please check that your frame images exist in the following directory:\n{image_folder}\n"
        "And that they are named sequentially, e.g., 'frame_00000.png', 'frame_00001.png', etc."
    )

# --- Save Final Tensors as .npy files ---
# Convert the list of tensors to a single tensor, then to a numpy array
depth_maps_array = torch.cat(depth_maps_all, dim=0).numpy()
# Convert the keypoint depth tensor to a numpy array
keypoint_depth_array = depth_tensor.numpy()

# Save results to the PROJECT_ROOT as .npy files
np.save(os.path.join(PROJECT_ROOT, "depth_map_cross_frames.npy"), depth_maps_array)
print(f"\nSaved per-frame depth maps as .npy: {depth_maps_array.shape}")

np.save(os.path.join(PROJECT_ROOT, "trowel_tip_keypoints_depth_tensor.npy"), keypoint_depth_array)
print(f"Saved keypoint depth tensor as .npy: {keypoint_depth_array.shape}")
