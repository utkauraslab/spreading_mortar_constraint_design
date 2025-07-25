import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# === Parameters ===
image_folder = "./bricklaying_data"  # Folder with frame_0000.png, etc.
output_folder = "./visualized_frames"
os.makedirs(output_folder, exist_ok=True)

# Load the precomputed 2D coordinates from the .pt tensor
coords_tensor = torch.load('trowel_tip_vertex_pred_tracks.pt')  # Shape: (3, 70, 2), integer coordinates

# Get dimensions
num_keypoints, num_frames, _ = coords_tensor.shape
print(f"Coords Tensor Shape: {coords_tensor.shape}")  # e.g., torch.Size([3, 70, 2])

# Assign unique colors to each keypoint using a colormap
cmap = plt.colormaps.get_cmap('tab20')
colors = (cmap(np.linspace(0, 1, num_keypoints))[:, :3] * 255).astype(np.uint8)

# Get image dimensions from the first frame (assuming all frames have same size)
first_frame_path = os.path.join(image_folder, "frame_0000.png")
if os.path.exists(first_frame_path):
    first_img = cv2.imread(first_frame_path)
    if first_img is None:
        raise FileNotFoundError(f"Could not load {first_frame_path}")
    h, w = first_img.shape[:2]
    print(f"Image dimensions: {w}x{h} pixels")
else:
    raise FileNotFoundError("No first frame found to determine image size")

# Visualize keypoints on each frame
for frame_idx in range(num_frames):
    img_path = os.path.join(image_folder, f"frame_{frame_idx:04d}.png")
    if not os.path.exists(img_path):
        print(f"Warning: Missing image {img_path}, skipping frame {frame_idx}")
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load {img_path}, skipping frame {frame_idx}")
        continue

    for kp_idx in range(num_keypoints):
        x, y = coords_tensor[kp_idx, frame_idx].tolist()
        # Check bounds to avoid drawing outside the image
        if 0 <= x < w and 0 <= y < h:
            color = tuple(map(int, colors[kp_idx]))
            cv2.circle(img, (x, y), radius=5, color=color, thickness=-1)  # Filled circle
            cv2.putText(img, f'{kp_idx+1}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Save the annotated image
    save_path = os.path.join(output_folder, f"frame_{frame_idx:04d}_visualized.png")
    cv2.imwrite(save_path, img)
    print(f"Saved visualized frame to {save_path}")

print(f"Visualized frames saved to {output_folder}")