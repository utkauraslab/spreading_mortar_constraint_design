"""
This script calculates the 6D pose (position and orientation) of the trowel
for each frame of the video. It uses the 2D polygon annotations and the refined
depth maps to create a 3D point cloud of the trowel for each frame. Then, it
applies Principal Component Analysis (PCA) to determine the trowel's local
coordinate frame and saves the resulting 4x4 homogeneous transformation matrix
for each frame.
"""

import os
import sys
import numpy as np
import pyvista as pv
import cv2
from tqdm import tqdm

# Camera intrinsic parameters
FX = 836.0
FY = 836.0
CX = 979.0
CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY])

# File paths (assuming the script is run from the project's root directory)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
OUTPUT_POSES_PATH = os.path.join(PROJECT_ROOT, "trowel_poses_trajectory.npy")

# Controls the size of the triangle representing the trowel.
TRIANGLE_EDGE_SIZE = 0.2  # in meters


def project_3D(coords_2d, depth_map, intrinsics):
    """Projects a list of 2D pixel coordinates into a 3D point cloud."""
    fx, fy, cx, cy = intrinsics
    if not isinstance(coords_2d, np.ndarray):
        coords_2d = np.array(coords_2d)
    x_coords, y_coords = coords_2d[:, 0], coords_2d[:, 1]
    depths = depth_map[y_coords, x_coords]
    valid_mask = (depths > 0) & ~np.isnan(depths)
    x_coords, y_coords, Z = x_coords[valid_mask], y_coords[valid_mask], depths[valid_mask]
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)


def calculate_local_frame(point_cloud):
    """
    PCA method
    Calculates the local coordinate frame (centroid and 3 orthogonal axes) for a point cloud.
    
    Returns:
        tuple: (centroid, x_axis, y_axis, z_axis(normal))
    """
    if point_cloud.shape[0] < 3:
        return None, None, None, None

    # get centroid point    
    centroid = np.mean(point_cloud, axis=0)

    # centralize
    centered_points = point_cloud - centroid

    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvectors by their corresponding eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # The axes correspond to the principal components (directions of variance)
    x_axis = eigenvectors[:, sorted_indices[0]] # Direction of most variance (length)
    y_axis = eigenvectors[:, sorted_indices[1]] # Direction of second-most variance (width)
    z_axis = eigenvectors[:, sorted_indices[2]] # Direction of least variance (the normal)
    z_axis = -z_axis
    return centroid, x_axis, y_axis, z_axis


def compute_pose_matrix(centroid, x_axis, y_axis, z_axis):
    """
    Constructs a 4x4 homogeneous transformation matrix from the local frame components.
    This matrix represents the transformation from the trowel's local frame to the camera's frame.
    """
    # Ensure all axes are normalized unit vectors
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Construct the 3x3 rotation matrix
    R = np.column_stack((x_axis, y_axis, z_axis))

    # Assemble the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = centroid

    return T





if __name__ == "__main__":
   
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    trowel_polygon_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)

    num_frames, height, width = all_depth_maps.shape

    
    # Calculate 3D point cloud in trowel region for each frame
    trowel_point_clouds_3d = []
    for i in range(num_frames):
        polygon_vertices_each_frame = trowel_polygon_vertices_2d_traj[i]
        if polygon_vertices_each_frame.size > 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_vertices_each_frame], 1)
            rows, cols = np.where(mask == 1)
            pixel_coords_2d = np.vstack((cols, rows)).T
            point_cloud_3d = project_3D(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
            trowel_point_clouds_3d.append(point_cloud_3d)
        else:
            trowel_point_clouds_3d.append(np.array([]))
    
    

    trowel_poses_trajectory = []
    for i in tqdm(range(num_frames), desc="Processing Poses"):
        point_cloud = trowel_point_clouds_3d[i]
        
        if point_cloud.size > 3:
            centroid, x_axis, y_axis, z_axis = calculate_local_frame(point_cloud)
            if centroid is not None:
                pose_matrix = compute_pose_matrix(centroid, x_axis, y_axis, z_axis)
                trowel_poses_trajectory.append(pose_matrix)
            else:
                # Append an identity matrix or NaN matrix if calculation fails
                trowel_poses_trajectory.append(np.full((4, 4), np.nan))
        else:
            # Append if there's no valid point cloud for the frame
            trowel_poses_trajectory.append(np.full((4, 4), np.nan))


    # Convert the list of matrices into a single NumPy array
    trowel_poses_trajectory = np.array(trowel_poses_trajectory)
    print(len(trowel_poses_trajectory))
    np.save(OUTPUT_POSES_PATH, trowel_poses_trajectory)



    # PyVista visualization setup
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # Segmented regions
    obj_names = ['brick_side_surface_by_sam_1', 
                 'brick_side_surface_by_sam_2', 
                 'brick_side_surface_by_sam_3', 
                 'brick_side_surface_by_sam_4',
                 'brick_side_surface_by_sam_5',
                 'ground_1', 
                 'ground_2']
    
    
    for name in obj_names:
        coords_2d_path = os.path.join(PROJECT_ROOT, 'seg_data', f'{name}.npy')
        if not os.path.exists(coords_2d_path): continue
        segmented_coords_2d = np.load(coords_2d_path)
        segmented_coords_3d = project_3D(segmented_coords_2d, 
                                               all_depth_maps[0], 
                                               INTRINSICS)
        if segmented_coords_3d.size > 0:
            plotter.add_points(segmented_coords_3d, 
                               style='points', 
                               color='#D95319', 
                               render_points_as_spheres=True, 
                               point_size=3)
    

    
    
    
    def update_trowel_frame(frame_value):
        frame_index = int(frame_value)
       
        actor_names = ["trowel_centroid", 
                       "trowel_x_axis", 
                       "trowel_y_axis", 
                       "trowel_z_axis", 
                       "trowel_triangle", 
                       "frame_text"]
        for name in actor_names:
            plotter.remove_actor(name)
        
        point_cloud = trowel_point_clouds_3d[frame_index]
        poses = []
        if point_cloud.size > 0:
            centroid, x_axis, y_axis, z_axis = calculate_local_frame(point_cloud)
            
            if centroid is not None:
                if frame_index == 39:
                    # print(f"Frame {frame_index + 1}: Flipping Z-axis direction.")
                    z_axis = -z_axis
                    y_axis = -y_axis

                # Visualize the centroid
                plotter.add_points(centroid, color='black', point_size=10, name="trowel_centroid")
                
                # Visualize the axes as arrows
                arrow_scale = 0.05
                plotter.add_arrows(cent=np.array([centroid]), direction=np.array([x_axis]), mag=arrow_scale, color='red', name="trowel_x_axis")
                plotter.add_arrows(cent=np.array([centroid]), direction=np.array([y_axis]), mag=arrow_scale, color='green', name="trowel_y_axis")
                plotter.add_arrows(cent=np.array([centroid]), direction=np.array([z_axis]), mag=arrow_scale, color='blue', name="trowel_z_axis")

                # Create and visualize an isosceles triangle
                length = TRIANGLE_EDGE_SIZE  # The length of the trowel representation
                width = TRIANGLE_EDGE_SIZE / 2 # The width of the base to make it pointier

                # Vertices of an isosceles triangle in 2D, with the point along the x-axis
                v1_2d = np.array([-length / 2, 0])      # Tip vertex
                v2_2d = np.array([length / 2, -width / 2]) # Back-left vertex
                v3_2d = np.array([length / 2, width / 2])  # Back-right vertex
                
                # Transform 2D vertices into the 3D local frame
                # The triangle is oriented along the local X (length) and Y (width) axes
                v1_3d = centroid + v1_2d[0] * x_axis + v1_2d[1] * y_axis
                v2_3d = centroid + v2_2d[0] * x_axis + v2_2d[1] * y_axis
                v3_3d = centroid + v3_2d[0] * x_axis + v3_2d[1] * y_axis
                
                triangle_points = np.array([v1_3d, v2_3d, v3_3d])
                face = np.hstack([3, 0, 1, 2])
                triangle_mesh = pv.PolyData(triangle_points, face)
                
                plotter.add_mesh(triangle_mesh, color='purple', opacity=0.8, name="trowel_triangle")

        plotter.add_text(f"Frame: {frame_index + 1}/{num_frames}", position='upper_edge', 
                         font_size=12, color='black', name="frame_text")
        return

    plotter.add_slider_widget(
        callback=update_trowel_frame,
        rng=[0, num_frames - 1], value=0, title="Frame", style='modern'
    )
    
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    
    update_trowel_frame(0)

    
    plotter.show()






# import os
# import sys
# import numpy as np
# import pyvista as pv
# import cv2

# # --- Configuration ---
# # Camera intrinsic parameters
# FX = 836.0
# FY = 836.0
# CX = 979.0
# CY = 632.0
# INTRINSICS = np.array([FX, FY, CX, CY])

# # File paths (assuming the script is run from the project's root directory)
# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
# TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")

# # --- Visualization Parameters ---
# # Controls the size of the triangle representing the trowel.
# TRIANGLE_EDGE_SIZE = 0.15  # in meters

# # --- Helper Functions ---

# def unproject_points(coords_2d, depth_map, intrinsics):
#     """Projects a list of 2D pixel coordinates into a 3D point cloud using individual depth values."""
#     fx, fy, cx, cy = intrinsics
#     if not isinstance(coords_2d, np.ndarray):
#         coords_2d = np.array(coords_2d)
#     x_coords, y_coords = coords_2d[:, 0], coords_2d[:, 1]
#     depths = depth_map[y_coords, x_coords]
#     valid_mask = (depths > 0) & ~np.isnan(depths)
#     x_coords, y_coords, Z = x_coords[valid_mask], y_coords[valid_mask], depths[valid_mask]
#     X = (x_coords - cx) * Z / fx
#     Y = (y_coords - cy) * Z / fy
#     return np.stack((X, Y, Z), axis=-1)

# def calculate_local_frame(point_cloud):
#     """
#     Calculates the local coordinate frame (centroid and 3 orthogonal axes) for a point cloud.
#     """
#     if point_cloud.shape[0] < 3:
#         return None, None, None, None
        
#     centroid = np.mean(point_cloud, axis=0)
#     centered_points = point_cloud - centroid
#     covariance_matrix = np.cov(centered_points, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
#     sorted_indices = np.argsort(eigenvalues)[::-1]
    
#     x_axis = eigenvectors[:, sorted_indices[0]] # Direction of most variance (length)
#     y_axis = eigenvectors[:, sorted_indices[1]] # Direction of second-most variance (width)
#     z_axis = eigenvectors[:, sorted_indices[2]] # Direction of least variance (the normal)
    
#     return centroid, x_axis, y_axis, z_axis

# # --- Main Execution ---
# if __name__ == "__main__":
#     print("Loading input data...")
#     all_depth_maps = np.load(DEPTH_MAP_PATH)
#     trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)

#     num_frames, height, width = all_depth_maps.shape

#     # --- 1. Pre-calculate all Trowel Point Clouds ---
#     print("Calculating 3D point cloud for the trowel in each frame...")
#     trowel_point_clouds_3d = []
#     for i in range(num_frames):
#         frame_vertices = trowel_vertices_2d_traj[i]
#         if frame_vertices.size > 0:
#             mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(mask, [frame_vertices], 1)
#             rows, cols = np.where(mask == 1)
#             pixel_coords_2d = np.vstack((cols, rows)).T
#             point_cloud_3d = unproject_points(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
#             trowel_point_clouds_3d.append(point_cloud_3d)
#         else:
#             trowel_point_clouds_3d.append(np.array([]))
    
#     print("All trowel point clouds calculated.")

#     # --- 2. PyVista Visualization Setup ---
#     plotter = pv.Plotter(window_size=[1200, 800])
#     plotter.set_background('white')

#     # --- 3. Plot Static Segmented Regions ---
#     obj_names = ['brick_side_surface_1', 'brick_side_surface_2', 
#                  'brick_side_surface_3', 'brick_side_surface_4',
#                  'brick_side_surface_5', 'ground_1', 'ground_2']
    
#     print("Projecting and plotting static background regions...")
#     for name in obj_names:
#         coords_2d_path = os.path.join(PROJECT_ROOT, 'seg_data', f'{name}.npy')
#         if not os.path.exists(coords_2d_path): continue
        
#         polygon_vertices_2d = np.load(coords_2d_path)
        
#         if polygon_vertices_2d.size > 0:
#             # *** UPDATED LOGIC: Always use the original depth for all static objects ***
#             # Create a mask to find all pixels inside the polygon
#             mask = np.zeros((height, width), dtype=np.uint8)
#             cv2.fillPoly(mask, [polygon_vertices_2d], 1)
#             rows, cols = np.where(mask == 1)
#             pixel_coords_inside_polygon = np.vstack((cols, rows)).T
            
#             # Unproject all pixels using their individual depth values from the first frame
#             segmented_coords_3d = unproject_points(pixel_coords_inside_polygon, all_depth_maps[0], INTRINSICS)

#             if segmented_coords_3d.size > 0:
#                 plotter.add_points(segmented_coords_3d, style='points', color='#D95319', 
#                                    render_points_as_spheres=True, point_size=3)
    
#     # --- 4. Interactive Trowel Local Frame Visualization ---
    
#     def update_trowel_frame(frame_value):
#         frame_index = int(frame_value)
#         actor_names = ["trowel_centroid", "trowel_x_axis", "trowel_y_axis", "trowel_z_axis", "trowel_triangle", "frame_text"]
#         for name in actor_names:
#             plotter.remove_actor(name)
        
#         point_cloud = trowel_point_clouds_3d[frame_index]
        
#         if point_cloud.size > 0:
#             centroid, x_axis, y_axis, z_axis = calculate_local_frame(point_cloud)
            
#             if centroid is not None:
#                 plotter.add_points(centroid, color='black', point_size=10, name="trowel_centroid")
#                 arrow_scale = 0.05
#                 plotter.add_arrows(cent=np.array([centroid]), direction=np.array([x_axis]), mag=arrow_scale, color='red', name="trowel_x_axis")
#                 plotter.add_arrows(cent=np.array([centroid]), direction=np.array([y_axis]), mag=arrow_scale, color='green', name="trowel_y_axis")
#                 plotter.add_arrows(cent=np.array([centroid]), direction=np.array([z_axis]), mag=arrow_scale, color='blue', name="trowel_z_axis")

#                 length = TRIANGLE_EDGE_SIZE
#                 width = TRIANGLE_EDGE_SIZE / 2
#                 v1_2d = np.array([-length, 0])
#                 v2_2d = np.array([0, -width / 2])
#                 v3_2d = np.array([0, width / 2])
                
#                 v1_3d = centroid + v1_2d[0] * x_axis + v1_2d[1] * y_axis
#                 v2_3d = centroid + v2_2d[0] * x_axis + v2_2d[1] * y_axis
#                 v3_3d = centroid + v3_2d[0] * x_axis + v3_2d[1] * y_axis
                
#                 triangle_points = np.array([v1_3d, v2_3d, v3_3d])
#                 face = np.hstack([3, 0, 1, 2])
#                 triangle_mesh = pv.PolyData(triangle_points, face)
                
#                 plotter.add_mesh(triangle_mesh, color='purple', opacity=0.8, name="trowel_triangle")

#         plotter.add_text(f"Frame: {frame_index + 1}/{num_frames}", position='upper_edge', 
#                          font_size=12, color='black', name="frame_text")
#         return

#     plotter.add_slider_widget(
#         callback=update_trowel_frame,
#         rng=[0, num_frames - 1], value=0, title="Frame", style='modern'
#     )
    
#     plotter.camera.azimuth = -60
#     plotter.camera.elevation = 25
#     plotter.camera.zoom(1.3)
    
#     update_trowel_frame(0)

#     print("Showing plot... Use the slider at the bottom to change frames.")
#     plotter.show()
