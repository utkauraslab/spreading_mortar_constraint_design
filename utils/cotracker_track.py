

# import torch
# import cv2
# import os
# from pathlib import Path
# from cotracker.predictor import CoTrackerPredictor
# from cotracker.utils.visualizer import Visualizer
# import numpy as np
# import json




# # Load video data
# video_path = './spreading_mortar_videos/mortar2.mp4'
# video = cv2.VideoCapture(video_path)
# if not video.isOpened():
#     raise FileNotFoundError(f"Could not open video: {video_path}")

# frames = []
# while True:
#     ret, frame = video.read()
#     if not ret:
#         break
#     frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
# video.release()

# video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()  # [1, C, H, W]
# num_frames = len(frames)  # Total number of frames (should be 70)

# # Move to GPU if available
# if torch.cuda.is_available():
#     video_tensor = video_tensor.cuda()

# # Load CoTracker model
# model = CoTrackerPredictor(
#     checkpoint=os.path.join('./co-tracker/checkpoints/scaled_offline.pth')
# )
# if torch.cuda.is_available():
#     model = model.cuda()

# # Load initial keypoints from JSON (frame 0 coordinates)
# with open("trowel_tip_keypoints.json", "r") as f:
#     points_data = json.load(f)

# # Convert to tensor: each row is [frame_index, x, y] for frame 0
# queries = torch.tensor([[0, float(p["x"]), float(p["y"])] for p in points_data])
# num_keypoints = len(points_data)

# if torch.cuda.is_available():
#     queries = queries.cuda()

# print(f"Initial Queries Shape: {queries.shape}")  # e.g., [3, 3]
# print(queries)

# # Run CoTracker inference
# pred_tracks, pred_visibility = model(video_tensor, queries=queries[None])  # [1, num_frames, num_keypoints, 2]
# print(f"Predicted Tracks Shape: {pred_tracks.shape}")  # Should be [1, 70, 3, 2]

# # Convert predicted tracks to integer coordinates and store in tensor
# coords_tensor = torch.zeros((num_keypoints, num_frames, 2), dtype=torch.int32)
# for f in range(num_frames):  # Loop over frames first
#     for k in range(num_keypoints):  # Then over keypoints
#         x, y = pred_tracks[0, f, k].cpu().numpy()  # Move to CPU if on GPU
#         x_int = int(round(x))  # Interpolate to nearest integer
#         y_int = int(round(y))  # Interpolate to nearest integer
#         # Clip to image bounds (assuming w=1958, h=1264)
#         x_int = np.clip(x_int, 0, 1957)
#         y_int = np.clip(y_int, 0, 1263)
#         coords_tensor[k, f] = torch.tensor([x_int, y_int], dtype=torch.int32)

# # Save the tensor with the correct shape
# torch.save(coords_tensor, "keypoints_2d_traj.pt") # 3 70, 2
# print(f"Saved predicted tracks to trowel_tip_keypoints_pred_tracks.pt, shape: {coords_tensor.shape}")  # [3, 70, 2]

# # Visualize and save the inference video
# output_video_dir = "./tracking_videos"
# os.makedirs(output_video_dir, exist_ok=True)
# vis = Visualizer(
#     save_dir=output_video_dir,
#     linewidth=6,
#     mode='cool',
#     tracks_leave_trace=-1  # Leave trace for all frames
# )
# vis.visualize(
#     video=video_tensor,
#     tracks=pred_tracks,
#     visibility=pred_visibility,
#     filename='trowel_tip_tracking'
# )
# print(f"Saved inference video to {output_video_dir}/trowel_tip_keypoints_tracking.mp4")













import torch
import cv2
import os
from pathlib import Path
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer
import numpy as np
import json
import sys

# --- Path Setup ---
# Establish the project root directory to ensure paths are correct
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(PROJECT_ROOT)

# --- Configuration ---
VIDEO_PATH = os.path.join(PROJECT_ROOT, 'spreading_mortar_videos/mortar2.mp4')
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'co-tracker/checkpoints/scaled_offline.pth')
KEYPOINTS_PATH = os.path.join(PROJECT_ROOT, "trowel_tip_keypoints.json")
OUTPUT_VIDEO_DIR = os.path.join(PROJECT_ROOT, "tracking_videos")
OUTPUT_TRAJECTORY_PATH = os.path.join(PROJECT_ROOT, "keypoints_2d_traj.npy")


# --- Main Execution ---
if __name__ == "__main__":
    # Load video data
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        sys.exit(1)
        
    video = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    video.release()

    if not frames:
        print("Error: No frames could be read from the video.")
        sys.exit(1)

    video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()
    num_frames = len(frames)

    # Move to GPU if available
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    video_tensor = video_tensor.to(DEVICE)

    # Load CoTracker model
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: CoTracker checkpoint not found at {CHECKPOINT_PATH}")
        print("Please run the 'setup_dependencies.sh' script to download it.")
        sys.exit(1)
        
    model = CoTrackerPredictor(checkpoint=CHECKPOINT_PATH)
    model = model.to(DEVICE)

    # Load initial keypoints from JSON (frame 0 coordinates)
    if not os.path.exists(KEYPOINTS_PATH):
        print(f"Error: Keypoints file not found at {KEYPOINTS_PATH}")
        sys.exit(1)

    with open(KEYPOINTS_PATH, "r") as f:
        points_data = json.load(f)

    # Convert to tensor: each row is [frame_index, x, y] for frame 0
    queries = torch.tensor([[0, float(p["x"]), float(p["y"])] for p in points_data], device=DEVICE)
    num_keypoints = len(points_data)

    print(f"Initial Queries Shape: {queries.shape}")

    # Run CoTracker inference
    pred_tracks, pred_visibility = model(video_tensor, queries=queries[None])
    print(f"Predicted Tracks Shape: {pred_tracks.shape}")

    # Convert predicted tracks to integer coordinates and store in tensor
    # The final tensor will be on the CPU
    coords_tensor = torch.zeros((num_keypoints, num_frames, 2), dtype=torch.int32)
    
    # Get image dimensions from the first frame for clipping
    H, W, _ = frames[0].shape
    
    for f in range(num_frames):
        for k in range(num_keypoints):
            x, y = pred_tracks[0, f, k].cpu().numpy()
            x_int = int(round(x))
            y_int = int(round(y))
            # Clip to image bounds
            x_int = np.clip(x_int, 0, W - 1)
            y_int = np.clip(y_int, 0, H - 1)
            coords_tensor[k, f] = torch.tensor([x_int, y_int], dtype=torch.int32)

    # --- Save the trajectory as a .npy file ---
    # Convert the final tensor to a NumPy array
    coords_array = coords_tensor.numpy()
    
    # Save the array using np.save
    np.save(OUTPUT_TRAJECTORY_PATH, coords_array)
    print(f"Saved predicted tracks to {OUTPUT_TRAJECTORY_PATH}, shape: {coords_array.shape}")

    # Visualize and save the inference video
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    vis = Visualizer(
        save_dir=OUTPUT_VIDEO_DIR,
        linewidth=6,
        mode='cool',
        tracks_leave_trace=-1  # Leave trace for all frames
    )
    vis.visualize(
        video=video_tensor,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename='trowel_tip_tracking'
    )
    print(f"Saved inference video to {os.path.join(OUTPUT_VIDEO_DIR, 'trowel_tip_tracking.mp4')}")
