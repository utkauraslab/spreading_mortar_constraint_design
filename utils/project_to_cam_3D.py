


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# # Load the 2D keypoints and depth values from .pt files
# # keypoints_2d = torch.load('coords_tensor.pt')  # Shape: (3, 70, 2)
# # depth_values = torch.load('keypoint_depth_tensor.pt')  # Shape: (3, 70)

# keypoints_2d = torch.load('keypoints_2d_traj.pt')  # Shape: (3, 70, 2)
# depth_values = torch.load('keypoints_depth_tensor.pt')  # Shape: (3, 70)

# # Convert to NumPy arrays for easier manipulation
# keypoints_2d = keypoints_2d.numpy()
# depth_values = depth_values.numpy()

# # Define camera intrinsic parameters
# fx = fy = 836
# cx = 979
# cy = 632

# # Project 2D points to 3D using camera intrinsics and depth
# num_keypoints, num_frames, _ = keypoints_2d.shape
# trajectories_3d = np.zeros((num_keypoints, num_frames, 3))

# for k in range(num_keypoints):
#     for f in range(num_frames):
#         u, v = keypoints_2d[k, f]
#         Z = depth_values[k, f]
#         X = (u - cx) * Z / fx
#         Y = (v - cy) * Z / fy
#         trajectories_3d[k, f] = [X, Y, Z]

# # Debug: Print sample coordinates to verify motion
# for k in range(num_keypoints): 
#     print(f"Keypoint {k+1} - Frame 0: {trajectories_3d[k, 0]}, Frame 69: {trajectories_3d[k, 69]}")


# print(trajectories_3d.shape)
# # torch.save(trajectories_3d, "trajectories_3d.pt")


# torch.save(trajectories_3d, "keypoints_3d_traj.pt")



# # Set up the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('3D Trajectories of Keypoints Across Frames')

# # Initialize scatter plots for each keypoint with different colors
# colors = ['red', 'green', 'blue']
# scatters = [ax.scatter([], [], [], c=colors[k], label=f'Keypoint {k+1}') for k in range(num_keypoints)]

# # Set plot limits with margin
# x_min, x_max = trajectories_3d[:, :, 0].min(), trajectories_3d[:, :, 0].max()
# y_min, y_max = trajectories_3d[:, :, 1].min(), trajectories_3d[:, :, 1].max()
# z_min, z_max = trajectories_3d[:, :, 2].min(), trajectories_3d[:, :, 2].max()

# ax.set_box_aspect((x_max-x_min, y_max-y_min, z_max-z_min))

# ax.set_xlim(x_min * 1.1, x_max * 1.1)
# ax.set_ylim(y_min * 1.1, y_max * 1.1)
# ax.set_zlim(z_min * 1.1, z_max * 1.1)

# #ax.view_init(elev=0, azim=-90)

# # Animation update function
# def update(frame):
#     for k in range(num_keypoints):
        
#         # scatters[k]._offsets3d = (trajectories_3d[k, :frame+1, 0], 
#         #                           trajectories_3d[k, :frame+1, 2], 
#         #                           trajectories_3d[k, :frame+1, 1])
        
#         xs = trajectories_3d[k, :frame+1, 0]          # horizontal
#         ys = trajectories_3d[k, :frame+1, 1]          # down
#         zs = trajectories_3d[k, :frame+1, 2]         # depth (out +)

#         scatters[k]._offsets3d = (xs, ys, zs)
#     ax.legend()
#     return scatters

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

# # Add legend and display
# plt.legend()
# plt.show()












# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation

# # Load the 2D keypoints and depth values from .pt files
# # keypoints_2d = torch.load('coords_tensor.pt')  # Shape: (3, 70, 2)
# # depth_values = torch.load('keypoint_depth_tensor.pt')  # Shape: (3, 70)

# keypoints_2d = torch.load('keypoints_2d_traj.pt')  # Shape: (3, 70, 2)
# depth_values = torch.load('keypoints_depth_tensor.pt')  # Shape: (3, 70)

# # Convert to NumPy arrays for easier manipulation
# keypoints_2d = keypoints_2d.numpy()
# depth_values = depth_values.numpy()

# # Define camera intrinsic parameters
# fx = fy = 836
# cx = 979
# cy = 632

# # Project 2D points to 3D using camera intrinsics and depth
# num_keypoints, num_frames, _ = keypoints_2d.shape
# trajectories_3d = np.zeros((num_keypoints, num_frames, 3))

# for k in range(num_keypoints):
#     for f in range(num_frames):
#         u, v = keypoints_2d[k, f]
#         Z = depth_values[k, f]
#         X = (u - cx) * Z / fx
#         Y = (v - cy) * Z / fy
#         trajectories_3d[k, f] = [X, Y, Z]

# # Debug: Print sample coordinates to verify motion
# for k in range(num_keypoints): 
#     print(f"Keypoint {k+1} - Frame 0: {trajectories_3d[k, 0]}, Frame 69: {trajectories_3d[k, 69]}")


# print(trajectories_3d.shape)
# # torch.save(trajectories_3d, "trajectories_3d.pt")


# torch.save(trajectories_3d, "keypoints_3d_traj.pt")



# # Set up the 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_zlabel('Z (m)')
# ax.set_title('3D Trajectories of Keypoints Across Frames')

# # Initialize scatter plots for each keypoint with different colors
# colors = ['red', 'green', 'blue']
# scatters = [ax.scatter([], [], [], c=colors[k], label=f'Keypoint {k+1}') for k in range(num_keypoints)]

# # Set plot limits with margin
# x_min, x_max = trajectories_3d[:, :, 0].min(), trajectories_3d[:, :, 0].max()
# y_min, y_max = trajectories_3d[:, :, 1].min(), trajectories_3d[:, :, 1].max()
# z_min, z_max = trajectories_3d[:, :, 2].min(), trajectories_3d[:, :, 2].max()

# ax.set_box_aspect((x_max-x_min, y_max-y_min, z_max-z_min))

# ax.set_xlim(x_min * 1.1, x_max * 1.1)
# ax.set_ylim(y_min * 1.1, y_max * 1.1)
# ax.set_zlim(z_min * 1.1, z_max * 1.1)

# #ax.view_init(elev=0, azim=-90)

# # Animation update function
# def update(frame):
#     for k in range(num_keypoints):
        
#         # scatters[k]._offsets3d = (trajectories_3d[k, :frame+1, 0], 
#         #                           trajectories_3d[k, :frame+1, 2], 
#         #                           trajectories_3d[k, :frame+1, 1])
        
#         xs = trajectories_3d[k, :frame+1, 0]          # horizontal
#         ys = trajectories_3d[k, :frame+1, 1]          # down
#         zs = trajectories_3d[k, :frame+1, 2]         # depth (out +)

#         scatters[k]._offsets3d = (xs, ys, zs)
#     ax.legend()
#     return scatters

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

# # Add legend and display
# plt.legend()
# plt.show()










import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# --- Configuration ---
# Define paths relative to the project root
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
KEYPOINTS_2D_PATH = os.path.join(PROJECT_ROOT, 'keypoints_2d_traj.npy')
DEPTH_VALUES_PATH = os.path.join(PROJECT_ROOT, 'trowel_tip_keypoints_depth_tensor.npy')
OUTPUT_3D_TRAJ_PATH = os.path.join(PROJECT_ROOT, 'keypoints_3d_traj.npy')

# --- Main Execution ---

# Load the 2D keypoints and depth values from .npy files
print("Loading data from .npy files...")
keypoints_2d = np.load(KEYPOINTS_2D_PATH)
depth_values = np.load(DEPTH_VALUES_PATH)

print(f"Loaded 2D keypoints with shape: {keypoints_2d.shape}")
print(f"Loaded depth values with shape: {depth_values.shape}")


# Define camera intrinsic parameters
fx = fy = 836.0
cx = 979.0
cy = 632.0

# Project 2D points to 3D using camera intrinsics and depth
num_keypoints, num_frames, _ = keypoints_2d.shape
trajectories_3d = np.zeros((num_keypoints, num_frames, 3))

for k in range(num_keypoints):
    for f in range(num_frames):
        u, v = keypoints_2d[k, f]
        Z = depth_values[k, f]
        
        # Only project if depth is valid (not NaN or zero)
        if Z > 0 and not np.isnan(Z):
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            trajectories_3d[k, f] = [X, Y, Z]
        else:
            # If depth is invalid, store as NaN
            trajectories_3d[k, f] = [np.nan, np.nan, np.nan]


# Debug: Print sample coordinates to verify motion
for k in range(num_keypoints): 
    print(f"Keypoint {k+1} - Frame 0: {trajectories_3d[k, 0]}, Frame {num_frames-1}: {trajectories_3d[k, num_frames-1]}")

# Save the 3D trajectories to a .npy file
np.save(OUTPUT_3D_TRAJ_PATH, trajectories_3d)
print(f"\nSaved 3D trajectories to {OUTPUT_3D_TRAJ_PATH} with shape: {trajectories_3d.shape}")


# # --- 3D Animation Setup ---
# print("Setting up 3D animation...")
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X (meters)')
# ax.set_ylabel('Y (meters)')
# ax.set_zlabel('Z (meters)')
# ax.set_title('3D Trajectories of Keypoints')

# # Initialize plots for each keypoint
# colors = ['red', 'green', 'blue']
# lines = [ax.plot([], [], [], '-', c=colors[k], label=f'Keypoint {k+1}')[0] for k in range(num_keypoints)]
# points = [ax.plot([], [], [], 'o', c=colors[k])[0] for k in range(num_keypoints)]


# # Set plot limits with a margin to avoid points being on the edge
# x_min, x_max = np.nanmin(trajectories_3d[:, :, 0]), np.nanmax(trajectories_3d[:, :, 0])
# y_min, y_max = np.nanmin(trajectories_3d[:, :, 1]), np.nanmax(trajectories_3d[:, :, 1])
# z_min, z_max = np.nanmin(trajectories_3d[:, :, 2]), np.nanmax(trajectories_3d[:, :, 2])

# margin_x = (x_max - x_min) * 0.1
# margin_y = (y_max - y_min) * 0.1
# margin_z = (z_max - z_min) * 0.1

# ax.set_xlim(x_min - margin_x, x_max + margin_x)
# ax.set_ylim(y_min - margin_y, y_max + margin_y)
# ax.set_zlim(z_min - margin_z, z_max + margin_z)

# # Set aspect ratio to be equal
# ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))


# # Animation update function
# def update(frame):
#     for k in range(num_keypoints):
#         # Get trajectory up to the current frame
#         xs = trajectories_3d[k, :frame+1, 0]
#         ys = trajectories_3d[k, :frame+1, 1]
#         zs = trajectories_3d[k, :frame+1, 2]
        
#         # Update the line plot (the full path)
#         lines[k].set_data(xs, ys)
#         lines[k].set_3d_properties(zs)
        
#         # Update the scatter plot (only the current point)
#         points[k].set_data(xs[-1:], ys[-1:])
#         points[k].set_3d_properties(zs[-1:])
        
#     # Set a dynamic title
#     ax.set_title(f'3D Trajectories of Keypoints (Frame {frame+1}/{num_frames})')
#     return lines + points

# # Create the animation
# ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

# # Add legend and display
# ax.legend()
# plt.show()





# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectories of Keypoints Across Frames')

# Initialize scatter plots for each keypoint with different colors
colors = ['red', 'green', 'blue']
scatters = [ax.scatter([], [], [], c=colors[k], label=f'Keypoint {k+1}') for k in range(num_keypoints)]

# Set plot limits with margin
x_min, x_max = trajectories_3d[:, :, 0].min(), trajectories_3d[:, :, 0].max()
y_min, y_max = trajectories_3d[:, :, 1].min(), trajectories_3d[:, :, 1].max()
z_min, z_max = trajectories_3d[:, :, 2].min(), trajectories_3d[:, :, 2].max()

ax.set_box_aspect((x_max-x_min, y_max-y_min, z_max-z_min))

ax.set_xlim(x_min * 1.1, x_max * 1.1)
ax.set_ylim(y_min * 1.1, y_max * 1.1)
ax.set_zlim(z_min * 1.1, z_max * 1.1)

#ax.view_init(elev=0, azim=-90)

# Animation update function
def update(frame):
    for k in range(num_keypoints):
        
        # scatters[k]._offsets3d = (trajectories_3d[k, :frame+1, 0], 
        #                           trajectories_3d[k, :frame+1, 2], 
        #                           trajectories_3d[k, :frame+1, 1])
        
        xs = trajectories_3d[k, :frame+1, 0]          # horizontal
        ys = trajectories_3d[k, :frame+1, 1]          # down
        zs = trajectories_3d[k, :frame+1, 2]         # depth (out +)

        scatters[k]._offsets3d = (xs, ys, zs)
    ax.legend()
    return scatters

# Create the animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=False)

# Add legend and display
plt.legend()
plt.show()
