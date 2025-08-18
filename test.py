"""
a static visualization of the 3D scene.
The static 3D point cloud of the entire brick wall's side surface and its best-fit plane.
The full trajectory of the trowel's centroid.

Also computes and saves:
  - T_wall2cam.npy         : (4,4)
  - T_trowel2cam.npy       : (N_valid, 4,4)   for the frames shown
  - T_trowel2brick.npy     : (N_valid, 4,4)   where T_trowel2brick[i] = inv(T_wall2cam) @ T_trowel2cam[i]
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
BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# --- Visualization Parameters ---

TRIANGLE_EDGE_SIZE = 0.15  # meters (long side of canonical triangle)
RECTANGLE_LENGTH   = 0.85  # meters
RECTANGLE_HEIGHT   = 0.05  # meters for the brick wall plane (band height)
RECTANGLE_WIDTH    = 0.16  # extrusion thickness along wall normal (visual only)


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
    (Uses eigenvectors; columns are orthonormal. We will only fix handedness later.)
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
    # Choose z_axis (normal) direction consistently: point towards the camera
    # (dot(z_axis, centroid) <= 0)
    if np.dot(z_axis, centroid) > 0:
        z_axis = -z_axis
    # Ensure right-handedness by flipping y_axis if necessary
    R_temp = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(R_temp) < 0:
        y_axis = -y_axis
    return centroid, x_axis, y_axis, z_axis

# --- SE(3) helpers ---

def se3_from_axes(origin, x_axis, y_axis, z_axis):
    """Build 4x4 pose T_cam_obj from axes (columns) and origin. Fix handedness if needed."""
    R = np.column_stack([x_axis, y_axis, z_axis])
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = origin
    return T

def invert_se3(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3,  3] = -R.T @ t
    return Ti


if __name__ == "__main__":
    
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    brick_wall_vertices_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)
    num_frames, height, width = all_depth_maps.shape

    # Pre-calculate all Trowel Local Frames 
    
    trowel_local_frames = []
    valid_frame_idx = []
    for i in tqdm(range(num_frames), desc="Calculating Trowel Frames"):
        frame_vertices = trowel_vertices_2d_traj[i]
        if frame_vertices is None or frame_vertices.size == 0:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [frame_vertices], 1)
        rows, cols = np.where(mask == 1)
        if rows.size == 0:
            continue
        pixel_coords_2d = np.vstack((cols, rows)).T
        point_cloud_3d = unproject_points(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
        if point_cloud_3d.size < 3:
            continue
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

    T_wall2cam = None
    T_trowel2cam_list = []

    if brick_wall_point_cloud_3d.size > 0:
        centroid, x_axis, y_axis, z_axis = calculate_local_frame(brick_wall_point_cloud_3d)
        if centroid is not None:
            # Flip y_axis to ensure trowel is above the wall in brick frame
            y_axis = -y_axis
            # Re-check and enforce right-handedness
            R_temp = np.column_stack((x_axis, y_axis, z_axis))
            if np.linalg.det(R_temp) < 0:
                x_axis = -x_axis  # Flip x to maintain handedness

            # --- Build T_wall->cam ---
            T_wall2cam = se3_from_axes(centroid, x_axis, y_axis, z_axis)
            #np.save(os.path.join(PROJECT_ROOT, "T_wall2cam.npy"), T_wall2cam)

            # --- Draw wall axes and box band like your original ---
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

    # --- Build T_trowel->cam for the frames you visualize ---
    if trowel_local_frames_view:
        for (centroid, x_axis, y_axis, z_axis) in trowel_local_frames_view:
            T_t = se3_from_axes(centroid, x_axis, y_axis, z_axis)
            T_trowel2cam_list.append(T_t)

        T_trowel2cam = np.stack(T_trowel2cam_list, axis=0)  # (Nv,4,4)
        #np.save(os.path.join(PROJECT_ROOT, "T_trowel2cam.npy"), T_trowel2cam)

        # --- If wall pose exists, compute T_trowel->brick per frame ---
        if T_wall2cam is not None:
            T_cam2brick = invert_se3(T_wall2cam)
            T_trowel2brick = T_cam2brick[None, ...] @ T_trowel2cam
            #np.save(os.path.join(PROJECT_ROOT, "T_trowel2brick.npy"), T_trowel2brick)
            #print(f"Saved {T_trowel2brick.shape[0]} relative poses to T_trowel2brick.npy")

    # --- Visualize trowel trajectory as before ---
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





    # === NEW: Visualize trowel trajectory in WALL (brick) frame ===
    if T_wall2cam is not None and len(T_trowel2cam_list) > 0:
        T_cam2wall = invert_se3(T_wall2cam)
        T_trowel2brick = T_cam2wall[None, ...] @ T_trowel2cam  # (Nv,4,4)

        # Extract positions and orientations in wall frame
        p_traj_brick = T_trowel2brick[:, :3, 3]
        R_brick_list = T_trowel2brick[:, :3, :3]

        plotter2 = pv.Plotter(window_size=[1200, 800])
        plotter2.set_background('white')

        # Brick-frame axes at origin
        origin = np.array([[0.0, 0.0, 0.0]])
        arrow_scale = 0.12
        plotter2.add_arrows(cent=origin, 
                            direction=np.array([[1,0,0]]), 
                            mag=arrow_scale, 
                            color='red',   
                            label='Brick X')
        plotter2.add_arrows(cent=origin, 
                            direction=np.array([[0,1,0]]), 
                            mag=arrow_scale, 
                            color='green', 
                            label='Brick Y')
        plotter2.add_arrows(cent=origin, 
                            direction=np.array([[0,0,1]]), 
                            mag=arrow_scale, 
                            color='blue',  
                            label='Brick Z')
        
        # plotter2.add_arrows(cent=origin, 
        #                        direction=np.array([x_axis]), 
        #                        mag=arrow_scale, 
        #                        color='red', 
        #                        label='Wall X-Axis')
        # plotter2.add_arrows(cent=origin, 
        #                        direction=np.array([y_axis]), 
        #                        mag=arrow_scale, 
        #                        color='green', 
        #                        label='Wall Y-Axis')
        # plotter2.add_arrows(cent=origin, 
        #                        direction=np.array([z_axis]), 
        #                        mag=arrow_scale, 
        #                        color='blue', 
        #                        label='Wall Z-Axis (Normal)')

        # Wall box at origin (brick frame): x=[1,0,0], y=[0,1,0], z=[0,0,1]
        l, w, h = RECTANGLE_LENGTH / 2, RECTANGLE_WIDTH / 2, RECTANGLE_HEIGHT / 2
        v1f = np.array([-l, -2*h, 0.0])
        v2f = np.array([ +l, -2*h, 0.0])
        v3f = np.array([ +l,  0.0, 0.0])
        v4f = np.array([-l,  0.0, 0.0])
        v1b = v1f + np.array([0.0, 0.0, w])
        v2b = v2f + np.array([0.0, 0.0, w])
        v3b = v3f + np.array([0.0, 0.0, w])
        v4b = v4f + np.array([0.0, 0.0, w])

        wall_pts_brick = np.vstack([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
        faces_brick = np.hstack([
            [4, 0,1,2,3],
            [4, 4,5,6,7],
            [4, 0,1,5,4],
            [4, 1,2,6,5],
            [4, 2,3,7,6],
            [4, 3,0,4,7],
        ]).astype(np.int64)
        plotter2.add_mesh(
            pv.PolyData(wall_pts_brick, faces_brick),
            color='#D95319', opacity=1.0,
            show_edges=True, edge_color='black', line_width=3,
            label='Wall Volume (brick)'
        )

        # Trowel path in BRICK frame
        plotter2.add_mesh(pv.Spline(p_traj_brick, 1000), color="blue", line_width=5, label="Trowel Path (brick)")
        plotter2.add_points(p_traj_brick[0],  color='green', point_size=15, render_points_as_spheres=True, label='Start (brick)')
        plotter2.add_points(p_traj_brick[-1], color='red',   point_size=15, render_points_as_spheres=True, label='End (brick)')

        # Per-frame oriented triangles in BRICK frame (using R_brick_list)
        for i, (R_bt, p_bt) in enumerate(zip(R_brick_list, p_traj_brick)):
            x_bt = R_bt[:, 0]
            y_bt = R_bt[:, 1]
            length, width = TRIANGLE_EDGE_SIZE, TRIANGLE_EDGE_SIZE / 2
            v1_2d = np.array([-length/2, 0.0])
            v2_2d = np.array([ length/2, -width/2])
            v3_2d = np.array([ length/2,  width/2])
            v1 = p_bt + v1_2d[0]*x_bt + v1_2d[1]*y_bt
            v2 = p_bt + v2_2d[0]*x_bt + v2_2d[1]*y_bt
            v3 = p_bt + v3_2d[0]*x_bt + v3_2d[1]*y_bt
            tri_pts = np.vstack([v1, v2, v3])
            face3 = np.hstack([3, 0, 1, 2]).astype(np.int64)
            opacity = (i + 1) / len(R_brick_list)
            plotter2.add_mesh(pv.PolyData(tri_pts, face3), color='grey', opacity=opacity, show_edges=True)

        plotter2.add_legend()
        plotter2.camera.azimuth = -60
        plotter2.camera.elevation = 25
        plotter2.camera.zoom(1.3)
        plotter2.show()