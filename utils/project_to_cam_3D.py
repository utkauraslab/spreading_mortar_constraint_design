


import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the 2D keypoints and depth values from .pt files
# keypoints_2d = torch.load('coords_tensor.pt')  # Shape: (3, 70, 2)
# depth_values = torch.load('keypoint_depth_tensor.pt')  # Shape: (3, 70)

keypoints_2d = torch.load('keypoints_2d_traj.pt')  # Shape: (3, 70, 2)
depth_values = torch.load('keypoints_depth_tensor.pt')  # Shape: (3, 70)

# Convert to NumPy arrays for easier manipulation
keypoints_2d = keypoints_2d.numpy()
depth_values = depth_values.numpy()

# Define camera intrinsic parameters
fx = fy = 836
cx = 979
cy = 632

# Project 2D points to 3D using camera intrinsics and depth
num_keypoints, num_frames, _ = keypoints_2d.shape
trajectories_3d = np.zeros((num_keypoints, num_frames, 3))

for k in range(num_keypoints):
    for f in range(num_frames):
        u, v = keypoints_2d[k, f]
        Z = depth_values[k, f]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        trajectories_3d[k, f] = [X, Y, Z]

# Debug: Print sample coordinates to verify motion
for k in range(num_keypoints): 
    print(f"Keypoint {k+1} - Frame 0: {trajectories_3d[k, 0]}, Frame 69: {trajectories_3d[k, 69]}")


print(trajectories_3d.shape)
# torch.save(trajectories_3d, "trajectories_3d.pt")


torch.save(trajectories_3d, "keypoints_3d_traj.pt")



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









