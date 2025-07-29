




# import os
# import torch
# import numpy as np
# from scipy.optimize import minimize
# from sklearn.neighbors import NearestNeighbors
# import sys
# import plotly.graph_objects as go

# # --- Configuration ---
# # Camera intrinsic parameters
# FX = 836.0
# FY = 836.0
# CX = 979.0
# CY = 632.0
# INTRINSICS = np.array([FX, FY, CX, CY])

# # File paths (assuming the script is run from the project root)
# SEGMENTED_COORDS_PATH = "segmented_pixel_coords.pt"
# DEPTH_MAP_PATH = "depth_map_cross_frames.pt"

# # Algorithm parameters
# NUM_PATCHES_TO_SAMPLE = 100  # How many local normals to calculate
# PATCH_SIZE = 50              # Number of points in each local patch
# INITIAL_GUESS_NORMAL = np.array([1.0, 1.0, 1.0]) # Initial guess for n (e.g., pointing up)

# # --- Helper Functions ---

# def unproject_points(coords_2d, depth_map, intrinsics):
#     """
#     Projects 2D pixel coordinates into a 3D point cloud.
#     """
#     fx, fy, cx, cy = intrinsics
#     points_3d = []
    
#     # Ensure coords_2d is a numpy array
#     if isinstance(coords_2d, torch.Tensor):
#         coords_2d = coords_2d.numpy()
        
#     for x, y in coords_2d:
#         # Pixel coordinates are integers
#         x, y = int(x), int(y)
#         if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
#             z = depth_map[y, x]
#             if z > 0:
#                 x_3d = (x - cx) * z / fx
#                 y_3d = (y - cy) * z / fy
#                 points_3d.append([x_3d, y_3d, z])
    
#     return np.array(points_3d)

# def calculate_patch_normal(patch_points):
#     """
#     Calculates the normal vector of a plane fitted to a patch of 3D points using PCA.
#     """
#     centroid = np.mean(patch_points, axis=0)
#     centered_points = patch_points - centroid
#     covariance_matrix = np.cov(centered_points, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
#     smallest_eigenvalue_index = np.argmin(eigenvalues)
#     normal = eigenvectors[:, smallest_eigenvalue_index]
#     return normal

# def objective_function(n, side_normals):
#     """
#     The function to be minimized.
#     """
#     n_unit = n / (np.linalg.norm(n) + 1e-9)
#     dot_products = np.abs(side_normals @ n_unit)
#     return np.sum(dot_products)

# def visualize_3d_scene(point_cloud, patches, side_normals, optimal_n):
#     """
#     Creates an interactive 3D plot of the point cloud, patches, and normal vectors.
#     """
#     print("Generating 3D visualization...")
#     fig = go.Figure()

#     # 1. Plot the full point cloud
#     fig.add_trace(go.Scatter3d(
#         x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
#         mode='markers',
#         marker=dict(size=2, color='lightgrey', opacity=0.7),
#         name='Full Side Surface'
#     ))

#     # 2. Plot the sampled patches
#     patch_points = np.vstack(patches)
#     fig.add_trace(go.Scatter3d(
#         x=patch_points[:, 0], y=patch_points[:, 1], z=patch_points[:, 2],
#         mode='markers',
#         marker=dict(size=3, color='cornflowerblue'),
#         name='Sampled Patches'
#     ))

#     # 3. Plot the side normals (si)
#     normal_scale = 0.01 # Scale factor to make normals visible
#     lines_x, lines_y, lines_z = [], [], []
#     for i, patch in enumerate(patches):
#         center = np.mean(patch, axis=0)
#         normal_end = center + side_normals[i] * normal_scale
#         lines_x.extend([center[0], normal_end[0], None])
#         lines_y.extend([center[1], normal_end[1], None])
#         lines_z.extend([center[2], normal_end[2], None])
    
#     fig.add_trace(go.Scatter3d(
#         x=lines_x, y=lines_y, z=lines_z,
#         mode='lines',
#         line=dict(color='cyan', width=3),
#         name='Side Normals (si)'
#     ))

#     # 4. Plot the optimized upper surface normal (n*)
#     cloud_centroid = np.mean(point_cloud, axis=0)
#     optimal_n_end = cloud_centroid + optimal_n * normal_scale * 5 # Make it bigger
#     fig.add_trace(go.Scatter3d(
#         x=[cloud_centroid[0], optimal_n_end[0]],
#         y=[cloud_centroid[1], optimal_n_end[1]],
#         z=[cloud_centroid[2], optimal_n_end[2]],
#         mode='lines',
#         line=dict(color='red', width=8),
#         name='Optimal Upper Normal (n*)'
#     ))

#     # Update layout for better viewing
#     fig.update_layout(
#         title="3D Visualization of Brick Surface and Normals",
#         scene=dict(
#             xaxis_title='X (meters)',
#             yaxis_title='Y (meters)',
#             zaxis_title='Z (meters)',
#             aspectmode='data' # Ensures correct aspect ratio
#         ),
#         margin=dict(l=0, r=0, b=0, t=40)
#     )
    
#     # Save to HTML and open in browser
#     fig.write_html("brick_surface_visualization.html")
#     print("\nSaved visualization to 'brick_surface_visualization.html'. Opening in browser...")
#     # Optional: automatically open the file
#     try:
#         import webbrowser
#         webbrowser.open("brick_surface_visualization.html")
#     except ImportError:
#         print("Could not open in browser automatically. Please open the HTML file manually.")


# # --- Main Execution ---

# if __name__ == "__main__":
#     # --- 1. Load Input Data ---
#     print("Loading input data...")
#     if not os.path.exists(SEGMENTED_COORDS_PATH):
#         print(f"Error: Segmented coordinates file not found at '{SEGMENTED_COORDS_PATH}'")
#         sys.exit(1)
#     if not os.path.exists(DEPTH_MAP_PATH):
#         print(f"Error: Depth map file not found at '{DEPTH_MAP_PATH}'")
#         sys.exit(1)

#     segmented_coords_2d = torch.load(SEGMENTED_COORDS_PATH)
#     depth_map_full = torch.load(DEPTH_MAP_PATH)[0].numpy()
    
#     print(f"Loaded {len(segmented_coords_2d)} segmented 2D points.")
#     print(f"Loaded depth map of shape {depth_map_full.shape}.")

#     # --- 2. Unproject to 3D Point Cloud ---
#     print("Unprojecting 2D points to 3D point cloud...")
#     point_cloud_3d = unproject_points(segmented_coords_2d, depth_map_full, INTRINSICS)
    
#     if len(point_cloud_3d) < PATCH_SIZE:
#         print(f"Error: Not enough valid 3D points ({len(point_cloud_3d)}) to form patches of size {PATCH_SIZE}.")
#         sys.exit(1)
        
#     print(f"Successfully created a 3D point cloud with {len(point_cloud_3d)} points.")

#     # --- 3. Sample Patches and Calculate Side Normals (si) ---
#     print(f"Sampling {NUM_PATCHES_TO_SAMPLE} patches to calculate side normals...")
    
#     random_indices = np.random.choice(len(point_cloud_3d), size=NUM_PATCHES_TO_SAMPLE, replace=False)
#     sample_centers = point_cloud_3d[random_indices]
    
#     nn = NearestNeighbors(n_neighbors=PATCH_SIZE, algorithm='auto').fit(point_cloud_3d)
#     distances, indices = nn.kneighbors(sample_centers)
    
#     side_normals = []
#     sampled_patches = []
#     for patch_indices in indices:
#         patch = point_cloud_3d[patch_indices]
#         sampled_patches.append(patch)
#         normal = calculate_patch_normal(patch)
#         side_normals.append(normal)
    
#     side_normals = np.array(side_normals)
#     print(f"Calculated {len(side_normals)} side normal vectors.")

#     # --- 4. Optimize for the Upper Surface Normal (n) ---
#     print("\nOptimizing for the upper surface normal vector 'n'...")
    
#     constraint = {'type': 'eq', 'fun': lambda n: np.linalg.norm(n) - 1.0}
    
#     print(f"Initial guess for n: {INITIAL_GUESS_NORMAL}")

#     result = minimize(
#         fun=objective_function,
#         x0=INITIAL_GUESS_NORMAL,
#         args=(side_normals,),
#         method='SLSQP',
#         constraints=[constraint],
#         options={'disp': True}
#     )

#     # --- 5. Display Results and Visualize ---
#     if result.success:
#         optimal_n = result.x
#         print("\n--- Optimization Successful ---")
#         print(f"Optimal upper surface normal vector (n*): {optimal_n}")
#         print(f"Final cost (sum of |cos(theta)|): {result.fun:.4f}")
#         print(f"Norm of the resulting vector: {np.linalg.norm(optimal_n):.4f}")
        
#         # Call the visualization function with the results
#         visualize_3d_scene(point_cloud_3d, sampled_patches, side_normals, optimal_n)
        
#     else:
#         print("\n--- Optimization Failed ---")
#         print(f"Message: {result.message}")








"""
PCA
"""


# import os
# import torch
# import numpy as np
# import sys
# import plotly.graph_objects as go

# # --- Configuration ---
# # Camera intrinsic parameters
# FX = 836.0
# FY = 836.0
# CX = 979.0
# CY = 632.0
# INTRINSICS = np.array([FX, FY, CX, CY])

# # File paths (assuming the script is run from the project root)
# SEGMENTED_COORDS_PATH = "segmented_pixel_coords.pt"
# DEPTH_MAP_PATH = "depth_map_cross_frames.pt"

# # An initial guess for the 'up' direction to correctly orient the final normal.
# # In many camera frames, -Y is up.
# GENERAL_UP_DIRECTION = np.array([0.0, 1.0, 0.0])

# # --- Helper Functions ---

# def unproject_points(coords_2d, depth_map, intrinsics):
#     """
#     Projects 2D pixel coordinates into a 3D point cloud.
#     """
#     fx, fy, cx, cy = intrinsics
#     points_3d = []
    
#     if isinstance(coords_2d, torch.Tensor):
#         coords_2d = coords_2d.numpy()
        
#     for x, y in coords_2d:
#         x, y = int(x), int(y)
#         if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
#             z = depth_map[y, x]
#             if z > 0:
#                 x_3d = (x - cx) * z / fx
#                 y_3d = (y - cy) * z / fy
#                 points_3d.append([x_3d, y_3d, z])
    
#     return np.array(points_3d)

# def calculate_side_surface_vectors(point_cloud):
#     """
#     Calculates the normal and primary direction vector of a point cloud using PCA.

#     Args:
#         point_cloud (np.ndarray): A set of 3D points, shape (N, 3).

#     Returns:
#         tuple: (surface_normal, edge_direction)
#     """
#     centroid = np.mean(point_cloud, axis=0)
#     centered_points = point_cloud - centroid
#     covariance_matrix = np.cov(centered_points, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
#     # Sort eigenvectors by their corresponding eigenvalues in descending order
#     sorted_indices = np.argsort(eigenvalues)[::-1]
    
#     # The primary edge direction corresponds to the LARGEST eigenvalue
#     edge_direction = eigenvectors[:, sorted_indices[0]]
    
#     # The surface normal corresponds to the SMALLEST eigenvalue
#     surface_normal = eigenvectors[:, sorted_indices[2]]
    
#     return surface_normal, edge_direction

# def visualize_3d_scene(point_cloud, side_normal, edge_direction, upper_normal):
#     """
#     Creates an interactive 3D plot of the point cloud and its principal vectors.
#     """
    
#     fig = go.Figure()

#     # 1. Plot the full point cloud
#     fig.add_trace(go.Scatter3d(
#         x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
#         mode='markers',
#         marker=dict(size=2, color='lightgrey', opacity=0.7),
#         name='Full Side Surface'
#     ))

#     # 2. Plot the principal vectors from the cloud's center
#     cloud_centroid = np.mean(point_cloud, axis=0)
#     vector_scale = 0.1 # Scale factor to make vectors visible

#     # Side Surface Normal (s_avg) - The vector perpendicular to the face
#     side_normal_end = cloud_centroid + side_normal * vector_scale
#     fig.add_trace(go.Scatter3d(
#         x=[cloud_centroid[0], side_normal_end[0]], y=[cloud_centroid[1], side_normal_end[1]], z=[cloud_centroid[2], side_normal_end[2]],
#         mode='lines', line=dict(color='cyan', width=5), name='Side Surface Normal'
#     ))

#     # Edge Direction (v_edge) - The vector along the longest dimension of the face
#     edge_dir_end = cloud_centroid + edge_direction * vector_scale
#     fig.add_trace(go.Scatter3d(
#         x=[cloud_centroid[0], edge_dir_end[0]], y=[cloud_centroid[1], edge_dir_end[1]], z=[cloud_centroid[2], edge_dir_end[2]],
#         mode='lines', line=dict(color='magenta', width=5), name='Edge Direction'
#     ))

#     # Optimal Upper Normal (n*) - The vector we want to find
#     upper_normal_end = cloud_centroid + upper_normal * vector_scale
#     fig.add_trace(go.Scatter3d(
#         x=[cloud_centroid[0], upper_normal_end[0]], y=[cloud_centroid[1], upper_normal_end[1]], z=[cloud_centroid[2], upper_normal_end[2]],
#         mode='lines', line=dict(color='red', width=8), name='Optimal Upper Normal (n*)'
#     ))
    
#     # Update layout for better viewing
#     fig.update_layout(
#         title="3D Visualization of Brick Surface and Principal Vectors",
#         scene=dict(
#             xaxis_title='X (meters)',
#             yaxis_title='Y (meters)',
#             zaxis_title='Z (meters)',
#             aspectmode='data' # Ensures correct aspect ratio
#         ),
#         margin=dict(l=0, r=0, b=0, t=40)
#     )
    
#     # Save to HTML and open in browser
#     fig.write_html("brick_surface_visualization.html")
#     print("\nSaved visualization to 'brick_surface_visualization.html'. Opening in browser...")
#     try:
#         import webbrowser
#         webbrowser.open("brick_surface_visualization.html")
#     except ImportError:
#         print("Could not open in browser automatically. Please open the HTML file manually.")


# # --- Main Execution ---

# if __name__ == "__main__":
#     # --- 1. Load Input Data ---
#     print("Loading input data...")
#     if not os.path.exists(SEGMENTED_COORDS_PATH):
#         print(f"Error: Segmented coordinates file not found at '{SEGMENTED_COORDS_PATH}'")
#         sys.exit(1)
#     if not os.path.exists(DEPTH_MAP_PATH):
#         print(f"Error: Depth map file not found at '{DEPTH_MAP_PATH}'")
#         sys.exit(1)

#     segmented_coords_2d = torch.load(SEGMENTED_COORDS_PATH)
#     depth_map_full = torch.load(DEPTH_MAP_PATH)[0].numpy()
    
#     print(f"Loaded {len(segmented_coords_2d)} segmented 2D points.")
#     print(f"Loaded depth map of shape {depth_map_full.shape}.")

#     # --- 2. Unproject to 3D Point Cloud ---
#     print("Unprojecting 2D points to 3D point cloud...")
#     point_cloud_3d = unproject_points(segmented_coords_2d, depth_map_full, INTRINSICS)
    
#     if len(point_cloud_3d) < 50: # Need enough points for stable PCA
#         print(f"Error: Not enough valid 3D points ({len(point_cloud_3d)}) to perform analysis.")
#         sys.exit(1)
        
#     print(f"Successfully created a 3D point cloud with {len(point_cloud_3d)} points.")

#     # --- 3. Calculate Principal Vectors of the Side Surface ---
#     print("Calculating principal vectors of the side surface using PCA...")
#     side_normal, edge_direction = calculate_side_surface_vectors(point_cloud_3d)

#     # --- 4. Calculate Upper Surface Normal via Cross Product ---
#     print("Calculating upper surface normal via cross product...")
#     # The upper normal is perpendicular to both the side normal and the edge direction
#     upper_normal_raw = np.cross(side_normal, edge_direction)
#     upper_normal_raw = -upper_normal_raw

#     # --- 5. Orient the Normal Vector Correctly ---
#     # Ensure the final normal points in the general "up" direction.
#     if np.dot(upper_normal_raw, GENERAL_UP_DIRECTION) < 0:
#         upper_normal_raw = -upper_normal_raw # Flip the vector if it's not pointing generally up
    
#     # Normalize the final vector to be a unit vector
#     optimal_n = upper_normal_raw / np.linalg.norm(upper_normal_raw)

#     # --- 6. Display Results and Visualize ---
#     print("\n--- Calculation Complete ---")
#     print(f"Side Surface Normal (s_avg): {side_normal}")
#     print(f"Edge Direction (v_edge):    {edge_direction}")
#     print(f"Optimal Upper Surface Normal (n*): {optimal_n}")

#     visualize_3d_scene(point_cloud_3d, side_normal, edge_direction, optimal_n)





"""
two loss
"""

import os
import torch
import numpy as np
from scipy.optimize import minimize
import sys
import plotly.graph_objects as go

# --- Configuration ---
# Camera intrinsic parameters
FX = 836.0
FY = 836.0
CX = 979.0
CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY])

# File paths (assuming the script is run from the project root)
SEGMENTED_COORDS_PATH = "segmented_pixel_coords.pt"
DEPTH_MAP_PATH = "depth_map_cross_frames.pt"

# --- USER INPUT: Define Vertical Lines ---
# Pairs of (x, y) coordinates representing vertical lines on the brick's side.
VERTICAL_LINE_PAIRS_2D = [
    [[298, 413], [298, 487]],
    [[339, 511], [339, 441]],
    [[456, 612], [456, 529]],
    [[528, 636], [528, 566]],
]

# --- Algorithm Parameters ---
# Weight for the combined loss function. alpha=1 uses only perpendicularity, alpha=0 uses only verticality.
ALPHA = 0.5 

# --- Helper Functions ---

def unproject_single_point(x, y, depth_map, intrinsics):
    """Unprojects a single 2D point to a 3D point."""
    fx, fy, cx, cy = intrinsics
    x, y = int(x), int(y)
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        z = depth_map[y, x]
        if z > 0:
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            return np.array([x_3d, y_3d, z])
    return None

def unproject_points(coords_2d, depth_map, intrinsics):
    """Projects a list of 2D pixel coordinates into a 3D point cloud."""
    points_3d = []
    if isinstance(coords_2d, torch.Tensor):
        coords_2d = coords_2d.numpy()
    for x, y in coords_2d:
        point_3d = unproject_single_point(x, y, depth_map, intrinsics)
        if point_3d is not None:
            points_3d.append(point_3d)
    return np.array(points_3d)

def calculate_side_surface_normal(point_cloud):
    """Calculates the single, general normal of a point cloud using PCA."""
    centroid = np.mean(point_cloud, axis=0)
    centered_points = point_cloud - centroid
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    smallest_eigenvalue_index = np.argmin(eigenvalues)
    return eigenvectors[:, smallest_eigenvalue_index]

def objective_function(n, side_normal, vertical_vectors, alpha):
    """
    Combined loss function.
    - Minimizes the angle between n and the vertical vectors.
    - Minimizes the angle between n and the plane of the side surface (i.e., maximizes angle with side_normal).
    """
    n_unit = n / (np.linalg.norm(n) + 1e-9)
    
    # Loss 1: Perpendicularity. We want dot product with side_normal to be 0.
    # Cost is |cos(theta_perp)|
    loss_perp = np.abs(np.dot(n_unit, side_normal))
    
    # Loss 2: Verticality. We want dot product with vertical vectors to be 1.
    # Cost is 1 - cos(theta_vertical)
    dot_products_vertical = vertical_vectors @ n_unit
    loss_vertical = np.sum(1 - dot_products_vertical)
    
    # Weighted combination of the two losses
    return alpha * loss_perp + (1 - alpha) * loss_vertical

def visualize_3d_scene(point_cloud, side_normal, vertical_lines_3d, optimal_n):
    """Creates an interactive 3D plot."""
    print("Generating 3D visualization...")
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
        mode='markers', marker=dict(size=2, color='lightgrey', opacity=0.5), name='Full Side Surface'
    ))

    cloud_centroid = np.mean(point_cloud, axis=0)
    vector_scale = 0.1

    # Plot Side Normal (s_avg)
    side_normal_end = cloud_centroid + side_normal * vector_scale
    fig.add_trace(go.Scatter3d(
        x=[cloud_centroid[0], side_normal_end[0]], y=[cloud_centroid[1], side_normal_end[1]], z=[cloud_centroid[2], side_normal_end[2]],
        mode='lines', line=dict(color='cyan', width=5), name='General Side Normal'
    ))

    # Plot Vertical Lines
    lines_x, lines_y, lines_z = [], [], []
    for start_point, end_point in vertical_lines_3d:
        lines_x.extend([start_point[0], end_point[0], None])
        lines_y.extend([start_point[1], end_point[1], None])
        lines_z.extend([start_point[2], end_point[2], None])
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='orange', width=4), name='Vertical Priors'
    ))

    # Plot Optimal Upper Normal (n*)
    optimal_n_end = cloud_centroid + optimal_n * vector_scale
    fig.add_trace(go.Scatter3d(
        x=[cloud_centroid[0], optimal_n_end[0]], y=[cloud_centroid[1], optimal_n_end[1]], z=[cloud_centroid[2], optimal_n_end[2]],
        mode='lines', line=dict(color='red', width=8), name='Optimal Upper Normal (n*)'
    ))
    
    fig.update_layout(
        title="3D Visualization with Combined Loss",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    fig.write_html("brick_surface_combined_loss.html")
    print("\nSaved visualization to 'brick_surface_combined_loss.html'. Opening in browser...")
    try:
        import webbrowser
        webbrowser.open("brick_surface_combined_loss.html")
    except ImportError:
        print("Could not open in browser automatically.")

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading input data...")
    if not os.path.exists(SEGMENTED_COORDS_PATH) or not os.path.exists(DEPTH_MAP_PATH):
        print("Error: Required data files not found.")
        sys.exit(1)

    segmented_coords_2d = torch.load(SEGMENTED_COORDS_PATH)
    depth_map_full = torch.load(DEPTH_MAP_PATH)[0].numpy()
    
    # --- 1. Unproject full side surface ---
    point_cloud_3d = unproject_points(segmented_coords_2d, depth_map_full, INTRINSICS)
    if len(point_cloud_3d) < 50:
        print("Error: Not enough valid 3D points for analysis.")
        sys.exit(1)
    
    # --- 2. Calculate the single, general side normal ---
    side_normal = calculate_side_surface_normal(point_cloud_3d)
    
    # --- 3. Create 3D vertical vectors from user input ---
    vertical_vectors_3d = []
    vertical_lines_for_vis = []
    for start_2d, end_2d in VERTICAL_LINE_PAIRS_2D:
        start_3d = unproject_single_point(start_2d[0], start_2d[1], depth_map_full, INTRINSICS)
        end_3d = unproject_single_point(end_2d[0], end_2d[1], depth_map_full, INTRINSICS)
        if start_3d is not None and end_3d is not None:
            direction_vec = end_3d - start_3d
            direction_vec /= np.linalg.norm(direction_vec)
            vertical_vectors_3d.append(direction_vec)
            vertical_lines_for_vis.append([start_3d, end_3d])

    if not vertical_vectors_3d:
        print("Error: Could not create any valid 3D vectors from the provided 2D pairs.")
        sys.exit(1)
    vertical_vectors_3d = np.array(vertical_vectors_3d)

    # --- 4. Optimize using the combined loss function ---
    print("\nOptimizing with combined loss function...")
    constraint = {'type': 'eq', 'fun': lambda n: np.linalg.norm(n) - 1.0}
    initial_guess = np.mean(vertical_vectors_3d, axis=0)
    initial_guess /= np.linalg.norm(initial_guess)
    
    print(f"Initial guess for n: {initial_guess}")
    print(f"Using loss weight alpha = {ALPHA}")

    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        args=(side_normal, vertical_vectors_3d, ALPHA),
        method='SLSQP',
        constraints=[constraint],
        options={'disp': True}
    )

    # --- 5. Display Results and Visualize ---
    if result.success:
        optimal_n = result.x
        print("\n--- Optimization Successful ---")
        print(f"Optimal upper surface normal vector (n*): {optimal_n}")
        visualize_3d_scene(point_cloud_3d, side_normal, vertical_lines_for_vis, optimal_n)
    else:
        print("\n--- Optimization Failed ---")
        print(f"Message: {result.message}")

