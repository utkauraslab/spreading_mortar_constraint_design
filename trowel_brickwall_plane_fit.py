# """
# This script analyzes the 3D geometry of the static brick wall.
# It loads the 2D polygon vertices for the entire wall, reconstructs its 3D
# point cloud, and then uses PCA to calculate the best-fit plane.

# The final visualization shows the raw point cloud along with the calculated
# centroid, local coordinate frame (axes), and a canonical rectangle representing
# the orientation of the fitted plane.
# """

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
# # Input file for the combined brick wall vertices
# BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# # --- Visualization Parameters ---
# RECTANGLE_LENGTH = 0.8  # in meters
# RECTANGLE_WIDTH = 0.1   # in meters

# # --- Helper Functions ---

# def unproject_points(coords_2d, depth_map, intrinsics):
#     """Projects a list of 2D pixel coordinates into a 3D point cloud."""
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
#     PCA method
#     Calculates the local coordinate frame (centroid and 3 orthogonal axes) for a point cloud.
    
#     Returns:
#         tuple: (centroid, x_axis, y_axis, z_axis(normal))
#     """
#     if point_cloud.shape[0] < 3:
#         return None, None, None, None

#     # get centroid point    
#     centroid = np.mean(point_cloud, axis=0)

#     # centralize
#     centered_points = point_cloud - centroid

#     covariance_matrix = np.cov(centered_points, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
#     # Sort eigenvectors by their corresponding eigenvalues in descending order
#     sorted_indices = np.argsort(eigenvalues)[::-1]
    
#     # The axes correspond to the principal components (directions of variance)
#     x_axis = eigenvectors[:, sorted_indices[0]] # Direction of most variance (length)
#     y_axis = eigenvectors[:, sorted_indices[1]] # Direction of second-most variance (width)
#     z_axis = eigenvectors[:, sorted_indices[2]] # Direction of least variance (the normal)
    
#     # Optional: Flip the normal if it's not pointing towards the camera
#     if z_axis[2] > 0:
#         z_axis = -z_axis

#     return centroid, x_axis, y_axis, z_axis

# # --- Main Execution ---
# if __name__ == "__main__":
#     print("Loading input data...")
#     depth_map_frame0 = np.load(DEPTH_MAP_PATH)[0]
#     brick_wall_vertices_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)

#     height, width = depth_map_frame0.shape

#     # --- 1. Create 3D Point Cloud for the Brick Wall ---
#     print("Creating 3D point cloud for the brick wall...")
#     brick_wall_mask = np.zeros((height, width), dtype=np.uint8)
#     cv2.fillPoly(brick_wall_mask, [brick_wall_vertices_2d], 1)
#     rows, cols = np.where(brick_wall_mask == 1)
#     brick_wall_pixels_2d = np.vstack((cols, rows)).T
#     brick_wall_point_cloud_3d = unproject_points(brick_wall_pixels_2d, depth_map_frame0, INTRINSICS)

#     if brick_wall_point_cloud_3d.size < 3:
#         print("Error: Not enough valid 3D points to analyze the brick wall.")
#         sys.exit(1)

#     # --- 2. Apply PCA to the Brick Wall Point Cloud ---
#     print("Applying PCA to find the best-fit plane for the brick wall...")
#     centroid, x_axis, y_axis, z_axis = calculate_local_frame(brick_wall_point_cloud_3d)

#     if centroid is None:
#         print("Error: PCA failed. Could not determine the local frame for the brick wall.")
#         sys.exit(1)

#     # --- 3. PyVista Visualization ---
#     plotter = pv.Plotter(window_size=[1200, 800])
#     plotter.set_background('white')

#     # Add the raw point cloud of the brick wall
#     plotter.add_points(brick_wall_point_cloud_3d, style='points', color='#D95319', 
#                        render_points_as_spheres=True, point_size=3, label='Brick Wall Point Cloud')

#     # Add the calculated centroid
#     plotter.add_points(centroid, color='black', point_size=10, label='Centroid')
    
#     # Add the calculated local frame axes as arrows
#     arrow_scale = 0.1
#     plotter.add_arrows(cent=np.array([centroid]), direction=np.array([x_axis]), mag=arrow_scale, color='red', label='X-Axis (Length)')
#     plotter.add_arrows(cent=np.array([centroid]), direction=np.array([y_axis]), mag=arrow_scale, color='green', label='Y-Axis (Width)')
#     plotter.add_arrows(cent=np.array([centroid]), direction=np.array([z_axis]), mag=arrow_scale, color='blue', label='Z-Axis (Normal)')

#     # Create and visualize a canonical rectangle oriented with the local frame
#     l = RECTANGLE_LENGTH / 2
#     w = RECTANGLE_WIDTH / 2
#     # Vertices of a rectangle in its own 2D plane
#     v1_2d, v2_2d, v3_2d, v4_2d = np.array([-l, -w]), np.array([l, -w]), np.array([l, w]), np.array([-l, w])
    
#     # Transform 2D vertices into the 3D local frame
#     v1_3d = centroid + v1_2d[0] * x_axis + v1_2d[1] * y_axis
#     v2_3d = centroid + v2_2d[0] * x_axis + v2_2d[1] * y_axis
#     v3_3d = centroid + v3_2d[0] * x_axis + v3_2d[1] * y_axis
#     v4_3d = centroid + v4_2d[0] * x_axis + v4_2d[1] * y_axis
    
#     rectangle_points = np.array([v1_3d, v2_3d, v3_3d, v4_3d])
#     face = np.hstack([4, 0, 1, 2, 3])
#     rectangle_mesh = pv.PolyData(rectangle_points, face)
    
#     plotter.add_mesh(rectangle_mesh, color='lightgreen', opacity=0.8, show_edges=True, edge_color='black', line_width=3, label='Best-Fit Plane')

#     plotter.add_legend()
#     plotter.camera.azimuth = -60
#     plotter.camera.elevation = 25
#     plotter.camera.zoom(1.3)
    
#     print("\nShowing plot... You can rotate and zoom with your mouse.")
#     plotter.show()











"""
visualization of the 3D scene:
1. The static 3D point cloud of the entire brick wall's side surface.
2. The best-fit plane and local coordinate frame for the static brick wall.
3. An interactive, frame-by-frame visualization of the trowel's local
   coordinate frame, represented by a canonical triangle.
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
# --- NEW: Path for the single brick wall file ---
BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

TROWEL_POSES_PATH = os.path.join(PROJECT_ROOT, "trowel_poses_trajectory.npy")
# --- Visualization Parameters ---
TRIANGLE_EDGE_SIZE = 0.15  # in meters for the trowel
RECTANGLE_LENGTH = 0.85     # in meters for the brick wall plane
RECTANGLE_HEIGHT = 0.1      # in meters for the brick wall plane
RECTANGLE_WIDTH = 0.08

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
    PCA method to calculate the local coordinate frame for a point cloud.
    """
    if point_cloud.shape[0] < 3:
        return None, None, None, None
    centroid = np.mean(point_cloud, axis=0)
    centered_points = point_cloud - centroid
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    x_axis = eigenvectors[:, sorted_indices[0]]
    y_axis = eigenvectors[:, sorted_indices[1]]
    z_axis = eigenvectors[:, sorted_indices[2]]
    z_axis = -z_axis
    return centroid, x_axis, y_axis, z_axis



# --- Main Execution ---
if __name__ == "__main__":
    print("Loading input data...")
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    brick_wall_vertices_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)

    num_frames, height, width = all_depth_maps.shape

    # --- 1. Pre-calculate all Trowel Point Clouds ---
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
    
    # --- 2. PyVista Visualization Setup ---
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # --- 3. Analyze and Plot Static Brick Wall ---
    print("Projecting and analyzing the static brick wall...")
    brick_wall_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(brick_wall_mask, [brick_wall_vertices_2d], 1)
    rows, cols = np.where(brick_wall_mask == 1)
    brick_wall_pixels_2d = np.vstack((cols, rows)).T
    brick_wall_point_cloud_3d = unproject_points(brick_wall_pixels_2d, all_depth_maps[0], INTRINSICS)

    if brick_wall_point_cloud_3d.size > 0:
        plotter.add_points(brick_wall_point_cloud_3d, style='points', color='#D95319', 
                           render_points_as_spheres=True, point_size=3, label='Brick Wall Cloud')
        
        centroid, x_axis, y_axis, z_axis = calculate_local_frame(brick_wall_point_cloud_3d)

        if centroid is not None:
            plotter.add_points(centroid, color='black', point_size=10, label='Wall Centroid')
            arrow_scale = 0.1
            plotter.add_arrows(cent=np.array([centroid]), direction=np.array([x_axis]), mag=arrow_scale, color='red', label='Wall X-Axis')
            plotter.add_arrows(cent=np.array([centroid]), direction=np.array([y_axis]), mag=arrow_scale, color='green', label='Wall Y-Axis')
            plotter.add_arrows(cent=np.array([centroid]), direction=np.array([z_axis]), mag=arrow_scale, color='blue', label='Wall Z-Axis (Normal)')

            # l, w = RECTANGLE_LENGTH / 2, RECTANGLE_HEIGHT / 2
            # v1_2d, v2_2d, v3_2d, v4_2d = np.array([-l, -w]), np.array([l, -w]), np.array([l, w]), np.array([-l, w])
            # v1_3d = centroid + v1_2d[0] * x_axis + v1_2d[1] * y_axis
            # v2_3d = centroid + v2_2d[0] * x_axis + v2_2d[1] * y_axis
            # v3_3d = centroid + v3_2d[0] * x_axis + v3_2d[1] * y_axis
            # v4_3d = centroid + v4_2d[0] * x_axis + v4_2d[1] * y_axis
            # rectangle_points = np.array([v1_3d, v2_3d, v3_3d, v4_3d])
            # face = np.hstack([4, 0, 1, 2, 3])
            # rectangle_mesh = pv.PolyData(rectangle_points, face)
            # plotter.add_mesh(rectangle_mesh, color='lightgreen', opacity=0.8, show_edges=True, edge_color='black', line_width=3, label='Wall Best-Fit Plane')


            box = pv.Box(bounds=(-RECTANGLE_LENGTH/2, 
                                 RECTANGLE_LENGTH/2, 
                                 -RECTANGLE_WIDTH*0.7, 
                                 RECTANGLE_WIDTH*0.3, 
                                 0, 
                                 RECTANGLE_WIDTH))
            
            # Create the 4x4 transformation matrix from the PCA results
            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3, 3] = centroid
            
            # Apply the transformation to the box
            box.transform(transform_matrix)
            
            plotter.add_mesh(box, color='lightgreen', opacity=0.5, show_edges=True, edge_color='black', line_width=2, label='Wall Best-Fit Box')


    # --- 4. Interactive Trowel Local Frame Visualization ---
    def update_trowel_frame(frame_value):
        frame_index = int(frame_value)
        # Define names for our dynamic actors so we can find and remove them
        actor_names = ["trowel_point_cloud", "trowel_centroid", "trowel_x_axis", 
                       "trowel_y_axis", "trowel_z_axis", "trowel_triangle", "frame_text"]
        for name in actor_names:
            plotter.remove_actor(name)
        
        point_cloud = trowel_point_clouds_3d[frame_index]
        
        if point_cloud.size > 0:
            
            # plotter.add_points(point_cloud, style='points', color='lightblue', 
            #                    render_points_as_spheres=True, point_size=3, name="trowel_point_cloud")

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



        plotter.add_text(f"Frame: {frame_index + 1}/{num_frames}", position='upper_edge', 
                         font_size=12, color='black', name="frame_text")
        return

    plotter.add_slider_widget(callback=update_trowel_frame, rng=[0, num_frames - 1], value=0, title="Frame", style='modern')
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    update_trowel_frame(0)
    plotter.add_legend()
    
    plotter.show()
