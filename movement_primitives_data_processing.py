"""
Prepare the position data for Prob-Movement Primitives to use.
stack with canonical time variable to serve as data:
    {x_t, y_{t}^{x}, y_{t}^{y}, y_{t}^{z}} for each time step
    x_t = (t-1) / (T-1) the cononical time variable(normalized time variable)

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the full trajectory of 4x4 pose matrices
t = np.load("trowel_poses_trajectory.npy")

print(f"Loaded poses shape: {t.shape}")
print(f"Data type: {t.dtype}")


num_extraction_frames = 70 
keypoint_traj_data = np.zeros((num_extraction_frames, 3), dtype=np.float64)

# Total number of timesteps for the canonical time variable
T = 70

# Loop through the desired frames (indices 30 to 69)
for i in range(0, 70):
    # Get the 4x4 pose matrix for the current frame
    pose_matrix = t[i]
    
    # Extract the translation vector (x, y, z position)
    translation_vector = pose_matrix[0:3, 3]
    
    # Store the translation vector in our new array.
    # We use i - 30 to map the frame index (30-69) to the array index (0-39).
    keypoint_traj_data[i] = translation_vector

print(f"Extracted keypoint trajectory shape: {keypoint_traj_data.shape}")



# Create the final data array with 4 columns (time, x, y, z)
demo_traj_data = np.zeros((num_extraction_frames, 4), dtype=np.float64)

for i in range(num_extraction_frames):
    # Calculate the canonical time variable for the current timestep
    # Note: Canonical time usually starts from 0, so we use `i / (T - 1)`
    x_t = i / (T - 1)
    
    # Get the 3D position for the current timestep
    position_vector = keypoint_traj_data[i]
    
    # Horizontally stack the time variable and the position vector
    # e.g., hstack( [0.0], [0.1, 0.2, 0.3] ) -> [0.0, 0.1, 0.2, 0.3]
    demo_traj_data[i] = np.hstack([x_t, position_vector])

# --- Verification ---
print("\nData preparation complete.")
print(f"Final demo trajectory shape: {demo_traj_data.shape}")
print(f"First row (t=0): {demo_traj_data[0]}")
print(f"Last row (t=1): {demo_traj_data[-1]}")


np.save('demo_keypoint_trajectory.npy', demo_traj_data)
print(demo_traj_data.shape)
# # --- 3D Visualization ---
# print("\nVisualizing the extracted 3D trajectory...")
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Extract X, Y, Z coordinates from the trajectory data
# x_coords = keypoint_traj_data[:, 0]
# y_coords = keypoint_traj_data[:, 1]
# z_coords = keypoint_traj_data[:, 2]

# # Plot the trajectory as a line
# ax.plot(x_coords, y_coords, z_coords, marker='o', linestyle='-', label='Trowel Path')

# # Mark the start and end points
# ax.scatter(x_coords[0], y_coords[0], z_coords[0], c='green', s=100, label='Start (Frame 0)')
# ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], c='red', s=100, label='End (Frame 69)')

# # Set labels and title
# ax.set_xlabel('X (meters)')
# ax.set_ylabel('Y (meters)')
# ax.set_zlabel('Z (meters)')
# ax.set_title('Extracted 3D Trowel Trajectory')
# ax.legend()
# ax.grid(True)

# # Set aspect ratio to be equal
# ax.set_box_aspect((np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)))

# plt.show()