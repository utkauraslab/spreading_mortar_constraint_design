# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation
# import cv2

# # Load the 2D keypoints and depth values from .pt files
# keypoints_2d = torch.load('coords_tensor.pt')  # Shape: (3, 70, 2)
# depth_values = torch.load('keypoint_depth_tensor.pt')  # Shape: (3, 70)

# # Convert to NumPy arrays for easier manipulation
# keypoints_2d = keypoints_2d.numpy()
# depth_values = depth_values.numpy()

# # Define camera intrinsic parameters and image height
# fx = fy = 2991
# cx = 979
# cy = 632
# h, w = 1264, 1958  # Image dimensions

# # Project 2D points to 3D using camera intrinsics and depth
# num_keypoints, num_frames, _ = keypoints_2d.shape
# trajectories_3d = np.zeros((num_keypoints, num_frames, 3))

# for k in range(num_keypoints):
#     for f in range(num_frames):
#         u, v = keypoints_2d[k, f]
#         Z = depth_values[k, f]
#         X = (u - cx) * Z / fx
#         Y = (h - 1 - v) * Z / fy  # y-axis: bottom to up
#         trajectories_3d[k, f] = [X, Y, Z]

# # Debug: Print sample coordinates to verify motion
# for k in range(num_keypoints):
#     print(f"Keypoint {k+1} - Frame 0: {trajectories_3d[k, 0]}, Frame 69: {trajectories_3d[k, 69]}")

# print(trajectories_3d.shape)
# torch.save(trajectories_3d, "trajectories_3d_reproject.pt")

# # Assume extrinsics from calibration (replace with actual rvecs, tvecs from calibrateCamera)
# rvec = np.array([0.1, 0.2, 0.3])  # Example rotation vector (radians)
# tvec = np.array([0.0, 0.0, 200.0])  # Example translation (mm), z=200 mm as initial depth
# R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to 3x3 matrix

# # Intrinsic matrix K
# K = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0, 0, 1]
# ], dtype=np.float32)

# # Reproject 3D points back to 2D
# reprojected_2d = np.zeros((num_keypoints, num_frames, 2))
# for f in range(num_frames):
#     for k in range(num_keypoints):
#         point_3d = trajectories_3d[k, f].reshape(1, 1, 3)
#         point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, K, None)
#         reprojected_2d[k, f] = point_2d[0, 0]

# # Set up the 3D plot with reprojection validation
# fig = plt.figure(figsize=(12, 5))

# # 3D Trajectory Plot
# ax3d = fig.add_subplot(121, projection='3d')
# ax3d.set_xlabel('X (mm) - Left to Right')
# ax3d.set_ylabel('Y (mm) - Bottom to Up')
# ax3d.set_zlabel('Z (mm) - Toward Picture')
# ax3d.set_title('3D Trajectories of Keypoints')
# ax3d.set_box_aspect([1, 1, 1])  # Equal scaling
# ax3d.view_init(elev=0, azim=180)  # z toward picture

# # Initialize 3D scatter plots
# colors = ['red', 'green', 'blue']
# scatters_3d = [ax3d.scatter([], [], [], c=colors[k], label=f'Keypoint {k+1}') for k in range(num_keypoints)]

# # 2D Reprojection Plot
# ax2d = fig.add_subplot(122)
# ax2d.set_xlabel('u (pixels)')
# ax2d.set_ylabel('v (pixels)')
# ax2d.set_title('Reprojected 2D Trajectories')
# ax2d.set_xlim(0, w)
# ax2d.set_ylim(h, 0)  # Invert y-axis to match image convention
# lines_2d = [ax2d.plot([], [], c=colors[k], label=f'Keypoint {k+1}')[0] for k in range(num_keypoints)]

# # Animation update function
# def update(frame):
#     for k in range(num_keypoints):
#         # Update 3D scatter
#         scatters_3d[k]._offsets3d = (trajectories_3d[k, :frame+1, 0], trajectories_3d[k, :frame+1, 1], trajectories_3d[k, :frame+1, 2])
#         # Update 2D line
#         lines_2d[k].set_data(reprojected_2d[k, :frame+1, 0], reprojected_2d[k, :frame+1, 1])
#     ax3d.legend()
#     ax2d.legend()
#     return scatters_3d + lines_2d

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

# # Display the plot
# plt.tight_layout()
# plt.show()

# # Optional: Save the animation (requires extra setup, e.g., PillowWriter)
# # from matplotlib.animation import PillowWriter
# # writer = PillowWriter(fps=10)
# # ani.save("3d_trajectories.gif", writer=writer)


















# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import cv2

# # Load the 2D keypoints and depth values from .pt files
# keypoints_2d = torch.load('coords_tensor.pt')  # Shape: (3, 70, 2)
# depth_values = torch.load('keypoint_depth_tensor.pt')  # Shape: (3, 70)

# # Convert to NumPy arrays for easier manipulation
# keypoints_2d = keypoints_2d.numpy()
# depth_values = depth_values.numpy()

# # Define camera intrinsic parameters and image dimensions
# fx = fy = 2991
# cx = 979
# cy = 632
# h, w = 1264, 1958  # Image height and width

# # Project 2D points to 3D using camera intrinsics and depth
# num_keypoints, num_frames, _ = keypoints_2d.shape
# trajectories_3d = np.zeros((num_keypoints, num_frames, 3))

# for k in range(num_keypoints):
#     for f in range(num_frames):
#         u, v = keypoints_2d[k, f]
#         Z = depth_values[k, f]
#         X = (u - cx) * Z / fx
#         Y = (h - 1 - v) * Z / fy  # y-axis: bottom to up for 3D, will be inverted for 2D reprojection
#         trajectories_3d[k, f] = [X, Y, Z]

# # Assume extrinsics from calibration (replace with actual rvecs[0], tvecs[0])
# rvec = np.array([0.1, 0.2, 0.3])  # Example rotation vector (radians)
# tvec = np.array([0.0, 0.0, 200.0])  # Example translation (mm), z=200 mm
# R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to 3x3 matrix

# # Intrinsic matrix K
# K = np.array([
#     [fx, 0, cx],
#     [0, fy, cy],
#     [0, 0, 1]
# ], dtype=np.float32)

# # Reproject 3D points back to 2D
# reprojected_2d = np.zeros((num_keypoints, num_frames, 2))
# for f in range(num_frames):
#     for k in range(num_keypoints):
#         point_3d = trajectories_3d[k, f].reshape(1, 1, 3)
#         point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, K, None)
#         reprojected_2d[k, f] = point_2d[0, 0]

# # Set up the 2D plot with top-left origin (y down)
# fig, ax = plt.subplots()
# ax.set_xlabel('X (pixels) - Left to Right')
# ax.set_ylabel('Y (pixels) - Top to Down')
# ax.set_title('Reprojected 2D Trajectories of Keypoints')
# ax.set_xlim(0, w)
# ax.set_ylim(h, 0)  # y increases downward from top-left origin

# # Initialize plot lines for each keypoint with different colors
# colors = ['red', 'green', 'blue']
# lines = [ax.plot([], [], c=colors[k], label=f'Keypoint {k+1}')[0] for k in range(num_keypoints)]

# # Animation update function
# def update(frame):
#     for k in range(num_keypoints):
#         lines[k].set_data(reprojected_2d[k, :frame+1, 0], reprojected_2d[k, :frame+1, 1])
#     ax.legend()
#     return lines

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

# # Add legend and display
# plt.legend()
# plt.show()

# # Optional: Compute reprojection error for validation
# reprojection_error = np.mean(np.linalg.norm(reprojected_2d - keypoints_2d, axis=2))
# print(f"Mean Reprojection Error: {reprojection_error:.2f} pixels")






















import torch
import numpy as np
import cv2
import os

# Load the 2D keypoints and depth values from .pt files
# keypoints_2d = torch.load('coords_tensor.pt')  # Shape: (3, 70, 2)

keypoints_2d = torch.load('trowel_tip_vertex_pred_tracks.pt')  # Shape: (3, 70, 2)
depth_values = torch.load('trowel_tip_vertex_depth_tensor.pt')  # Shape: (3, 70)

# Convert to NumPy arrays for easier manipulation
keypoints_2d = keypoints_2d.numpy()
depth_values = depth_values.numpy()

# Define camera intrinsic parameters and image dimensions
fx = fy = 836
cx = 979
cy = 632
h, w = 1264, 1958  # Image height and width

# Project 2D points to 3D using camera intrinsics and depth
num_keypoints, num_frames, _ = keypoints_2d.shape
trajectories_3d = np.zeros((num_keypoints, num_frames, 3))

for k in range(num_keypoints):
    for f in range(num_frames):
        u, v = keypoints_2d[k, f]
        Z = depth_values[k, f]
        X = (u - cx) * Z / fx
        #Y = (h - 1 - v) * Z / fy  # y-axis: bottom to up for 3D
        Y = (v - cy) * Z / fy
        trajectories_3d[k, f] = [X, Y, Z]





# Assume extrinsics from calibration (replace with actual rvecs[0], tvecs[0])
# rvec = np.array([0.48138572, 0.85104319, 0.22976743])  
# tvec = np.array([-103.39510327, -6.97140344, 802.67947447]) 

# rvec = np.array([1.28769252, -0.90302623, 1.89943465])  # rotation vector (radians)
# tvec = np.array([62.46543879, 9.62128031, -94.12325778])  # translation

rvec = np.zeros((3,1), dtype=np.float32)  # rotation vector (radians)
tvec = np.zeros((3,1), dtype=np.float32)  # translation
R, _ = cv2.Rodrigues(rvec)  # Convert rotation vector to 3x3 matrix

# Intrinsic matrix K
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)

# Reproject 3D points back to 2D
reprojected_2d = np.zeros((num_keypoints, num_frames, 2))
for f in range(num_frames):
    for k in range(num_keypoints):
        point_3d = trajectories_3d[k, f].reshape(1, 1, 3)
        point_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, K, None)
        reprojected_2d[k, f] = point_2d[0, 0]

# Create output directory for annotated frames
output_dir = "./backproj_annotated_frames"
os.makedirs(output_dir, exist_ok=True)

# Load and annotate each frame
image_folder = "./bricklaying_data"
for f in range(num_frames):
    frame_path = os.path.join(image_folder, f"frame_{f:04d}.png")
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Warning: Could not load {frame_path}, skipping frame {f}")
        continue
    
    # Draw reprojected points
    for k in range(num_keypoints):
        u, v = reprojected_2d[k, f].astype(int)
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img, (u, v), 5, (0, 0, 255), -1)  # Red circle
            cv2.putText(img, f'{k+1}', (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save annotated image


    
    output_path = os.path.join(output_dir, f"frame_{f:04d}_annotated.png")
    cv2.imwrite(output_path, img)
    print(f"Saved annotated frame to {output_path}")

# Optional: Compute reprojection error for validation
reprojection_error = np.mean(np.linalg.norm(reprojected_2d - keypoints_2d, axis=2))
print(f"Mean Reprojection Error: {reprojection_error:.2f} pixels")