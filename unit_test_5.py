"""
a static visualization of the 3D scene.
The static 3D point cloud of the entire brick wall's side surface and its best-fit plane.
The full trajectory of the trowel's centroid.

"""

import os
import sys
import numpy as np
import pyvista as pv
import cv2
from tqdm import tqdm


FX = 836.0
FY = 836.0
CX = 979.0
CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY])


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
TROWEL_TIP_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_tip_polygon_vertices.npy")
BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# --- Visualization Parameters ---

TRIANGLE_EDGE_SIZE = 0.15  # meters (long side of canonical triangle)
RECTANGLE_LENGTH   = 0.85  # meters
RECTANGLE_HEIGHT = 0.05      # in meters for the brick wall plane
RECTANGLE_WIDTH = 0.16


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



if __name__ == "__main__":
    
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    trowel_tip_vertices_2d_traj = np.load(TROWEL_TIP_VERTICES_2D_PATH, allow_pickle=True)
    brick_wall_vertices_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)
    print(trowel_vertices_2d_traj.shape)
    print(trowel_tip_vertices_2d_traj.shape)
    num_frames, height, width = all_depth_maps.shape

    # Pre-calculate all Trowel Local Frames 
    
    trowel_local_frames = []
    valid_frame_idx = []
    for i in tqdm(range(num_frames), desc="Calculating Trowel Frames"):
        frame_vertices = trowel_vertices_2d_traj[i]
        if frame_vertices.size > 0:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [frame_vertices], 1)
            rows, cols = np.where(mask == 1)
            pixel_coords_2d = np.vstack((cols, rows)).T
            point_cloud_3d = unproject_points(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
            
            local_frame = calculate_local_frame(point_cloud_3d)
            if local_frame[0] is not None:
                trowel_local_frames.append(local_frame)
                valid_frame_idx.append(i)

    START_AT_FRAME = 20
    keep_pos = [k for k, fidx in enumerate(valid_frame_idx) if fidx >= (START_AT_FRAME-1)]
    trowel_local_frames_view = [trowel_local_frames[k] for k in keep_pos]

    # 2 PyVista Visualization Setup ---
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # 3 Analyze and Plot Static Brick Wall ---
    brick_wall_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(brick_wall_mask, [brick_wall_vertices_2d], 1)
    rows, cols = np.where(brick_wall_mask == 1)
    brick_wall_pixels_2d = np.vstack((cols, rows)).T
    brick_wall_point_cloud_3d = unproject_points(brick_wall_pixels_2d, all_depth_maps[0], INTRINSICS)

    if brick_wall_point_cloud_3d.size > 0:
        
        centroid, x_axis, y_axis, z_axis = calculate_local_frame(brick_wall_point_cloud_3d)

        if centroid is not None:
            plotter.add_points(centroid, color='black', point_size=10, label='Wall Centroid')
            arrow_scale = 0.05
            plotter.add_arrows(cent=np.array([centroid]), 
                               direction=np.array([x_axis]), 
                               mag=arrow_scale, 
                               color='red', 
                               label='Wall X-Axis')
            plotter.add_arrows(cent=np.array([centroid]), 
                               direction=np.array([y_axis]), 
                               mag=arrow_scale, 
                               color='green', 
                               label='Wall Y-Axis')
            plotter.add_arrows(cent=np.array([centroid]), 
                               direction=np.array([z_axis]), 
                               mag=arrow_scale, 
                               color='blue', 
                               label='Wall Z-Axis (Normal)')

            l, w, h = RECTANGLE_LENGTH / 2, RECTANGLE_WIDTH / 2, RECTANGLE_HEIGHT / 2
            v1_2d, v2_2d, v3_2d, v4_2d = np.array([-l, -2*h]), np.array([l, -2*h]), np.array([l, 0]), np.array([-l, 0])
            
            v1f = centroid + v1_2d[0]*x_axis + v1_2d[1]*y_axis
            v2f = centroid + v2_2d[0]*x_axis + v2_2d[1]*y_axis
            v3f = centroid + v3_2d[0]*x_axis + v3_2d[1]*y_axis
            v4f = centroid + v4_2d[0]*x_axis + v4_2d[1]*y_axis

            v1b = centroid + v1_2d[0]*x_axis + v1_2d[1]*y_axis + w*z_axis
            v2b = centroid + v2_2d[0]*x_axis + v2_2d[1]*y_axis + w*z_axis
            v3b = centroid + v3_2d[0]*x_axis + v3_2d[1]*y_axis + w*z_axis
            v4b = centroid + v4_2d[0]*x_axis + v4_2d[1]*y_axis + w*z_axis

            
            pts = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])

            faces = np.hstack([
                [4, 0,1,2,3],   # front
                [4, 4,5,6,7],   # back
                [4, 0,1,5,4],   # side
                [4, 1,2,6,5],   # side
                [4, 2,3,7,6],   # side
                [4, 3,0,4,7],   # side
            ]).astype(np.int64)

            box_mesh = pv.PolyData(pts, faces)
            plotter.add_mesh(
                box_mesh,
                color='#D95319',
                opacity=1,
                show_edges=True,
                edge_color='black',
                line_width=3,
                label='Wall Volume'
            )


    if trowel_local_frames_view:
        trowel_centroid_trajectory = np.array([frame[0] for frame in trowel_local_frames_view])
        plotter.add_mesh(pv.Spline(trowel_centroid_trajectory, 1000), color="blue", line_width=5, label="Trowel Centroid Path")
        plotter.add_points(trowel_centroid_trajectory[0], color='green', point_size=15, render_points_as_spheres=True, label='Start')
        plotter.add_points(trowel_centroid_trajectory[-1], color='red', point_size=15, render_points_as_spheres=True, label='End')

        for i, (centroid, x_axis, y_axis, z_axis) in enumerate(trowel_local_frames_view):
            length, width = TRIANGLE_EDGE_SIZE, TRIANGLE_EDGE_SIZE / 2
            v1_2d = np.array([-length/2, 0])
            v2_2d = np.array([ length/2, -width/2])
            v3_2d = np.array([ length/2,  width/2])

            v1_3d = centroid + v1_2d[0]*x_axis + v1_2d[1]*y_axis
            v2_3d = centroid + v2_2d[0]*x_axis + v2_2d[1]*y_axis
            v3_3d = centroid + v3_2d[0]*x_axis + v3_2d[1]*y_axis

            triangle_points = np.array([v1_3d, v2_3d, v3_3d])
            face = np.hstack([3, 0, 1, 2])
            triangle_mesh = pv.PolyData(triangle_points, face)

            opacity = (i + 1) / len(trowel_local_frames_view)
            plotter.add_mesh(triangle_mesh, 
                             color='grey', 
                             opacity=0.8, 
                             show_edges=True)

    plotter.add_legend()
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    plotter.show()
