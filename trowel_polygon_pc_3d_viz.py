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
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT,  "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")

# --- Helper Functions ---

def unproject_points(coords_2d, depth_map, intrinsics):
    """Projects a list of 2D pixel coordinates into a 3D point cloud."""
    fx, fy, cx, cy = intrinsics
    points_3d = []
    
    # Ensure coords_2d is a numpy array
    if not isinstance(coords_2d, np.ndarray):
        coords_2d = np.array(coords_2d)

    # Extract x and y columns
    x_coords = coords_2d[:, 0]
    y_coords = coords_2d[:, 1]
    
    # Get depth values for all coordinates at once for efficiency
    depths = depth_map[y_coords, x_coords]
    
    # Filter out invalid depths
    valid_mask = (depths > 0) & ~np.isnan(depths)
    
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    Z = depths[valid_mask]
    
    # Perform unprojection in a vectorized way
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    
    # Stack the coordinates into a (N, 3) array
    return np.stack((X, Y, Z), axis=-1)





if __name__ == "__main__":
    print("Loading input data...")
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)

    num_frames, height, width = all_depth_maps.shape

   
    trowel_point_clouds_3d = []
    for i in range(num_frames):
        frame_vertices = trowel_vertices_2d_traj[i]
        
        if frame_vertices.size > 0:
            # Create a black image (mask) of the same size as the frame
            mask = np.zeros((height, width), dtype=np.uint8)
            # Fill the polygon defined by the vertices with white
            cv2.fillPoly(mask, [frame_vertices], 1)
            
            # Find the (y, x) coordinates of all pixels inside the polygon
            rows, cols = np.where(mask == 1)

            # Combine into an (N, 2) array of (x, y) coordinates
            pixel_coords_2d = np.vstack((cols, rows)).T
            
            # Unproject these pixels to a 3D point cloud
            point_cloud_3d = unproject_points(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
            trowel_point_clouds_3d.append(point_cloud_3d)
        else:
            # If no vertices, add an empty point cloud for this frame
            trowel_point_clouds_3d.append(np.array([]))
    
    

    # --- 2. PyVista Visualization Setup ---
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # --- 3. Plot Static Segmented Regions ---
    obj_names = ['brick_side_surface_1', 'brick_side_surface_2', 
                 'brick_side_surface_3', 'ground_1', 'ground_2']
    
    print("Projecting and plotting static background regions...")
    for name in obj_names:
        coords_2d_path = os.path.join(PROJECT_ROOT, 'seg_data', f'{name}.npy')
        if not os.path.exists(coords_2d_path):
            continue
        segmented_coords_2d = np.load(coords_2d_path)
        segmented_coords_3d = unproject_points(segmented_coords_2d, all_depth_maps[0], INTRINSICS)
        if segmented_coords_3d.size > 0:
            plotter.add_points(segmented_coords_3d, style='points', color='#D95319', 
                               render_points_as_spheres=True, point_size=3)
    
    # --- 4. Interactive Trowel Visualization ---
    
    def update_trowel_cloud(frame_value):
        frame_index = int(frame_value)
        actor_name = "trowel_cloud"
        text_actor_name = "frame_text"
        
        plotter.remove_actor(actor_name)
        plotter.remove_actor(text_actor_name)
        
        point_cloud = trowel_point_clouds_3d[frame_index]
        
        if point_cloud.size > 0:
            plotter.add_points(point_cloud, style='points', color="lightblue", 
                               render_points_as_spheres=True, point_size=3, name=actor_name)
        
        plotter.add_text(f"Frame: {frame_index + 1}/{num_frames}", position='upper_edge', 
                         font_size=12, color='black', name=text_actor_name)
        return

    plotter.add_slider_widget(
        callback=update_trowel_cloud,
        rng=[0, num_frames - 1],
        value=0,
        title="Frame",
        style='modern'
    )
    
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    
    update_trowel_cloud(0)

    print("Showing plot... Use the slider at the bottom to change frames.")
    plotter.show()
