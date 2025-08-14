


# import os
# import sys
# import torch
# import numpy as np
# import pyvista as pv


# # Cam intrinsic

# FX = 836.0
# FY = 836.0
# CX = 979.0
# CY = 632.0

# INTRINSICS = np.array([FX, FY, CX, CY])


# # file path

# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# # DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "seg_data/depth_map_cross_frames.npy")
# DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
# # TIP_3D_TRAJ_PATH = os.path.join(PROJECT_ROOT, "seg_data/keypoints_3d_traj.npy")

# TIP_3D_TRAJ_PATH = os.path.join(PROJECT_ROOT, "keypoints_3d_traj_refined.npy")



# def project_single_point(x, y, depth_map, intrinsics):
#     fx, fy, cx, cy = intrinsics
#     x, y = int(x), int(y)

#     if 0<=y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
#         z = depth_map[y, x]
#         if z > 0 and not np.isnan(z):
#             x_3d = (x-cx) * z / fx
#             y_3d = (y-cy) * z / fy
#             return np.array([x_3d, y_3d, z])
#     return None


# def project_points(coords_2d, depth_map, intrinsics):
#     points_3d = []
#     if not isinstance(coords_2d, np.ndarray):
#         coords_2d = np.array(coords_2d)

#     for x, y in coords_2d:
#         point_3d = project_single_point(x, y, depth_map, intrinsics)
#         if points_3d is not None:
#             points_3d.append(point_3d)
#     return np.array(points_3d)



# if __name__=="__main__":
#     depth_map_full = np.load(DEPTH_MAP_PATH)[0]
#     tip_traj_3d = np.load(TIP_3D_TRAJ_PATH)

#     # Pyvista visualization
#     plotter = pv.Plotter(window_size=[1200, 800])
#     plotter.set_background('white')


#     obj_name = ['brick_side_surface_1', 
#                 'brick_side_surface_2', 
#                 'brick_side_surface_3', 
#                 'ground_1', 'ground_2', 
#                 'trowel_region']
    

#     for i in range(len(obj_name)):
#         segmented_coords_2d = np.load(PROJECT_ROOT + '/seg_data/' + obj_name[i] + '.npy')
#         segmented_coords_3d = project_points(segmented_coords_2d, 
#                                              depth_map_full, 
#                                              INTRINSICS)
        

#         plotter.add_points(
#             segmented_coords_3d,
#             style='points',
#             color='#D95319',
#             render_points_as_spheres=True,
#             point_size=5,
#             emissive=True
#         )
    


#     # Add the trowel tip trajectory as a series of meshes to create a trail
#     num_frames = tip_traj_3d.shape[1]
#     for i in range(num_frames):
#         if not np.isnan(tip_traj_3d[:, i, :]).any():
#             face_array = np.hstack([3, 0, 1, 2])
#             trowel_mesh = pv.PolyData(tip_traj_3d[:, i, :], face_array)


#             # Use opacity to create a fading trail effect
#             opacity = (i+1) / num_frames
#             plotter.add_mesh(trowel_mesh, show_edges=True, color="lightblue", opacity=opacity, line_width=2)

    
#     plotter.show()

































# import os
# import sys
# import numpy as np
# import pyvista as pv

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
# TROWEL_VERTICES_3D_TRAJ_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices_3d.npy")

# # --- Helper Functions ---

# def unproject_single_point(x, y, depth_map, intrinsics):
#     """Unprojects a single 2D point to a 3D point."""
#     fx, fy, cx, cy = intrinsics
#     x, y = int(x), int(y)
#     if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
#         z = depth_map[y, x]
#         if z > 0 and not np.isnan(z):
#             x_3d = (x - cx) * z / fx
#             y_3d = (y - cy) * z / fy
#             return np.array([x_3d, y_3d, z])
#     return None

# def unproject_points(coords_2d, depth_map, intrinsics):
#     """Projects a list of 2D pixel coordinates into a 3D point cloud."""
#     points_3d = []
#     if not isinstance(coords_2d, np.ndarray):
#         coords_2d = np.array(coords_2d)

#     for x, y in coords_2d:
#         point_3d = unproject_single_point(x, y, depth_map, intrinsics)
#         if point_3d is not None:
#             points_3d.append(point_3d)
#     return np.array(points_3d)

# # --- Main Execution ---
# if __name__ == "__main__":
#     print("Loading input data...")
#     # Load the refined depth map (we only need the first frame for static objects)
#     depth_map_frame0 = np.load(DEPTH_MAP_PATH)[0]
#     # Load the 3D trowel trajectory (which is an array of arrays)
#     trowel_vertices_3d_traj = np.load(TROWEL_VERTICES_3D_TRAJ_PATH, allow_pickle=True)

#     # --- PyVista Visualization Setup ---
#     plotter = pv.Plotter(window_size=[1200, 800])
#     plotter.set_background('white')

#     # --- Plot Static Segmented Regions ---
#     # List of segmented object names
#     obj_names = ['brick_side_surface_1', 
#                  'brick_side_surface_2', 
#                  'brick_side_surface_3', 
#                  'ground_1', 
#                  'ground_2', ]
#                 #  'trowel_region']
    
#     print("Projecting and plotting static segmented regions...")
#     for i, name in enumerate(obj_names):
#         coords_2d_path = os.path.join(PROJECT_ROOT, 'seg_data', f'{name}.npy')
#         if not os.path.exists(coords_2d_path):
#             print(f"Warning: File not found for '{name}', skipping.")
#             continue
            
#         segmented_coords_2d = np.load(coords_2d_path)
#         segmented_coords_3d = unproject_points(segmented_coords_2d, depth_map_frame0, INTRINSICS)
        
#         if segmented_coords_3d.size > 0:
#             plotter.add_points(
#                 segmented_coords_3d,
#                 style='points',
#                 # *** UPDATED: Use a single color for all static regions ***
#                 color='#D95319', 
#                 render_points_as_spheres=True,
#                 point_size=5,
#                 emissive=True,
#                 label=name 
#             )
    
#     # --- Plot Trowel Trajectory Trail ---
#     print("Plotting trowel trajectory trail...")
#     num_frames = len(trowel_vertices_3d_traj)
    
#     for i in range(num_frames):
#         # Get the vertex array for the current frame
#         frame_vertices = trowel_vertices_3d_traj[i]
        
#         # Check if the frame has valid, non-NaN data
#         if frame_vertices.size > 0 and not np.isnan(frame_vertices).any():
#             # Create a face that connects all vertices in order
#             num_vertices = len(frame_vertices)
#             face_array = np.hstack([num_vertices, np.arange(num_vertices)])
#             trowel_mesh = pv.PolyData(frame_vertices, face_array)

#             # Use opacity to create a fading trail effect
#             opacity = (i + 1) / num_frames
            
#             # Only add the label for the last frame to avoid duplicates in the legend
#             trowel_label = "Trowel Trajectory" if i == num_frames - 1 else None
            
#             plotter.add_mesh(
#                 trowel_mesh, 
#                 # *** UPDATED: Changed 'wireframe' to 'surface' to show a solid object ***
#                 style='surface', 
#                 color="lightblue", 
#                 opacity=opacity,
#                 label=trowel_label
#             )
    
#     # Add a legend to identify the objects
#     plotter.add_legend()
#     plotter.camera.azimuth = -60
#     plotter.camera.elevation = 25
#     plotter.camera.zoom(1.3)

#     print("Showing plot... You can rotate and zoom with your mouse.")
#     plotter.show()

























import os
import sys
import numpy as np
import pyvista as pv

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
TROWEL_VERTICES_3D_TRAJ_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices_3d.npy")

# --- Helper Functions ---

def unproject_single_point(x, y, depth_map, intrinsics):
    """Unprojects a single 2D point to a 3D point."""
    fx, fy, cx, cy = intrinsics
    x, y = int(x), int(y)
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        z = depth_map[y, x]
        if z > 0 and not np.isnan(z):
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            return np.array([x_3d, y_3d, z])
    return None

def unproject_points(coords_2d, depth_map, intrinsics):
    """Projects a list of 2D pixel coordinates into a 3D point cloud."""
    points_3d = []
    if not isinstance(coords_2d, np.ndarray):
        coords_2d = np.array(coords_2d)

    for x, y in coords_2d:
        point_3d = unproject_single_point(x, y, depth_map, intrinsics)
        if point_3d is not None:
            points_3d.append(point_3d)
    return np.array(points_3d)

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading input data...")
    depth_map_frame0 = np.load(DEPTH_MAP_PATH)[0]
    trowel_vertices_3d_traj = np.load(TROWEL_VERTICES_3D_TRAJ_PATH, allow_pickle=True)

    # --- PyVista Visualization Setup ---
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # --- Plot Static Segmented Regions ---
    obj_names = ['brick_side_surface_1', 
                 'brick_side_surface_2', 
                 'brick_side_surface_3', 
                 'brick_side_surface_4',
                 'brick_side_surface_5',
                 'ground_1', 
                 'ground_2']
    
    print("Projecting and plotting static segmented regions...")
    for name in obj_names:
        coords_2d_path = os.path.join(PROJECT_ROOT, 'seg_data', f'{name}.npy')
        if not os.path.exists(coords_2d_path):
            print(f"Warning: File not found for '{name}', skipping.")
            continue
            
        segmented_coords_2d = np.load(coords_2d_path)
        segmented_coords_3d = unproject_points(segmented_coords_2d, depth_map_frame0, INTRINSICS)
        
        if segmented_coords_3d.size > 0:
            plotter.add_points(
                segmented_coords_3d,
                style='points',
                color='#D95319', 
                render_points_as_spheres=True,
                point_size=5,
                emissive=True,
            )
    
    # --- Interactive Trowel Visualization ---
    num_frames = len(trowel_vertices_3d_traj)
    
    # This function will be called every time the slider is moved
    def update_trowel_mesh(frame_value):
        frame_index = int(frame_value)
        
        # Define names for our dynamic actors so we can find and remove them
        trowel_actor_name = "trowel_mesh"
        text_actor_name = "frame_text"
        
        # Remove the previous actors to clear the scene for the new frame
        plotter.remove_actor(trowel_actor_name)
        plotter.remove_actor(text_actor_name)
        
        # Get the vertex array for the current frame
        frame_vertices = trowel_vertices_3d_traj[frame_index]
        
        # *** CORRECTED LOGIC: Convert to float before checking for NaN ***
        # This prevents the TypeError with np.isnan
        if isinstance(frame_vertices, np.ndarray) and frame_vertices.size > 0:
            frame_vertices_float = frame_vertices.astype(np.float64)
        
            # Check if the frame has valid, non-NaN data
            if not np.isnan(frame_vertices_float).any():
                num_vertices = len(frame_vertices_float)
                face_array = np.hstack([num_vertices, np.arange(num_vertices)])
                trowel_mesh = pv.PolyData(frame_vertices_float, face_array)
                
                # Add the new trowel mesh for the current frame
                plotter.add_mesh(
                    trowel_mesh, 
                    style='surface', 
                    color="lightblue",
                    show_edges=True,
                    name=trowel_actor_name # Assign the name here
                )
        
        # Add the new text for the current frame
        plotter.add_text(
            f"Frame: {frame_index + 1}/{num_frames}", 
            position='upper_edge', 
            font_size=12, 
            color='black',
            name=text_actor_name # Assign the name here
        )
        return

    # Add the slider widget to the plotter
    plotter.add_slider_widget(
        callback=update_trowel_mesh,
        rng=[0, num_frames - 1], # Range of the slider
        value=0, # Initial value
        title="Frame",
        style='modern'
    )
    
    # Set the initial view
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    
    # Manually call the update function once to show the first frame
    update_trowel_mesh(0)

    print("Showing plot... Use the slider at the bottom to change frames.")
    plotter.show()
