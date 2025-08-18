"""


This script visualizes the ground, brick side surfaces, trowel polygon region 3D point clouds under camera frame.
Visualzie the local frame and cononical triangle shape under local frame which represent's trowel tip region.
Also transformed triangle vertices under local frame into camera frame to verify the pose trajectory extraction process.

"""


import os
import sys
import numpy as np
import pyvista as pv
import cv2

# --- Configuration ---
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
TROWEL_POSES_PATH = os.path.join(PROJECT_ROOT, "trowel_poses_trajectory.npy")

# --- Visualization Parameters ---
# Controls the size of the triangle representing the trowel.
TRIANGLE_EDGE_SIZE = 0.15  # in meters

# --- Helper Functions ---

def unproject_points(coords_2d, depth_map, intrinsics):
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
    Calculates the local coordinate frame (centroid and 3 orthogonal axes) for a point cloud.
    
    Returns:
        tuple: (centroid, x_axis, y_axis, z_axis(normal))
    """
    if point_cloud.shape[0] < 3:
        return None, None, None, None
        
    centroid = np.mean(point_cloud, axis=0)
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




if __name__ == "__main__":
    
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)

    num_frames, height, width = all_depth_maps.shape

    # Pre-calculate all Trowel Point Clouds
    print("Calculating 3D point cloud for the trowel in each frame...")
    trowel_point_clouds_3d = []
    for i in range(num_frames):
        frame_vertices = trowel_vertices_2d_traj[i]
        if frame_vertices.size > 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [frame_vertices], 1)
            rows, cols = np.where(mask == 1)
            pixel_coords_2d = np.vstack((cols, rows)).T
            point_cloud_3d = unproject_points(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
            trowel_point_clouds_3d.append(point_cloud_3d)
        else:
            trowel_point_clouds_3d.append(np.array([]))
    
    

    # PyVista Visualization Setup
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # Plot Static Segmented Regions
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
        segmented_coords_3d = unproject_points(segmented_coords_2d, all_depth_maps[0], INTRINSICS)
        if segmented_coords_3d.size > 0:
            plotter.add_points(segmented_coords_3d, style='points', color='#D95319', 
                               render_points_as_spheres=True, point_size=3)
    
    # Interactive trowel local frame and canonical triangle shape visualization
    def update_trowel_frame(frame_value):
        frame_index = int(frame_value)
        # Define names for our dynamic actors so we can find and remove them
        actor_names = ["trowel_point_cloud", "trowel_centroid", "trowel_x_axis", 
                       "trowel_y_axis", "trowel_z_axis", "trowel_triangle", "frame_text"]
        for name in actor_names:
            plotter.remove_actor(name)
        
        point_cloud = trowel_point_clouds_3d[frame_index]
        
        if point_cloud.size > 0:
            # *** NEW: Add the raw point cloud visualization ***
            plotter.add_points(point_cloud, style='points', color='lightblue', 
                               render_points_as_spheres=True, point_size=3, name="trowel_point_cloud")

            centroid, x_axis, y_axis, z_axis = calculate_local_frame(point_cloud)
            
            if centroid is not None:
                # Special condition for the 40th frame (index 39)
                if frame_index == 39:
                    z_axis = -z_axis
                    y_axis = -y_axis

                # Visualize the centroid
                plotter.add_points(centroid, color='black', point_size=10, name="trowel_centroid")
                
                # Visualize the axes as arrows
                arrow_scale = 0.05
                plotter.add_arrows(cent=np.array([centroid]), direction=np.array([x_axis]), mag=arrow_scale, color='red', name="trowel_x_axis")
                plotter.add_arrows(cent=np.array([centroid]), direction=np.array([y_axis]), mag=arrow_scale, color='green', name="trowel_y_axis")
                plotter.add_arrows(cent=np.array([centroid]), direction=np.array([z_axis]), mag=arrow_scale, color='blue', name="trowel_z_axis")

                # Create and visualize a centered isosceles triangle
                length = TRIANGLE_EDGE_SIZE
                width = TRIANGLE_EDGE_SIZE / 2

                v1_2d = np.array([-length / 2, 0])      # Tip vertex
                v2_2d = np.array([length / 2, -width / 2]) # Back-left vertex
                v3_2d = np.array([length / 2, width / 2])  # Back-right vertex
                
                # Transform 2D vertices into the 3D local frame
                v1_3d = centroid + v1_2d[0] * x_axis + v1_2d[1] * y_axis
                v2_3d = centroid + v2_2d[0] * x_axis + v2_2d[1] * y_axis
                v3_3d = centroid + v3_2d[0] * x_axis + v3_2d[1] * y_axis
                
                triangle_points = np.array([v1_3d, v2_3d, v3_3d])
                face = np.hstack([3, 0, 1, 2])
                triangle_mesh = pv.PolyData(triangle_points, face)
                
                plotter.add_mesh(triangle_mesh, color='purple', opacity=0.8, name="trowel_triangle")

                # === Verification Using Transformation Matrix ===
                pose_traj = np.load(TROWEL_POSES_PATH)  # shape: (num_frames, 4, 4)
                T = pose_traj[frame_index]  # (4,4)

                # Local frame triangle vertices (homogeneous)
                v1_local = np.array([-length / 2, 0, 0, 1.0])
                v2_local = np.array([length / 2, -width / 2, 0, 1.0])
                v3_local = np.array([length / 2, width / 2, 0, 1.0])

                # Transform to camera frame using T
                v1_cam = (T @ v1_local)[:3]
                v2_cam = (T @ v2_local)[:3]
                v3_cam = (T @ v3_local)[:3]

                # Add transformed triangle vertices as colored dots
                plotter.add_points(v1_cam, color='yellow', point_size=12, name="T_point1")
                plotter.add_points(v2_cam, color='yellow', point_size=12, name="T_point2")
                plotter.add_points(v3_cam, color='yellow', point_size=12, name="T_point3")



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
