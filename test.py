"""
Visualize everything in the brick-wall local frame.

Inputs (same as before):
- depth_map_cross_frames_refined.npy    : (N, H, W) depth maps
- trowel_polygon_vertices.npy           : list/array of per-frame polygon vertices in image pixels (x,y), may be empty for some frames
- brick_wall_side_surface.npy           : single polygon vertices of the wall side surface (x,y) in a reference frame (use frame 0 depth)
Optional:
- You can save T_wall2cam computed here if you want future reuse; but always keep visualization and transforms consistent.

Coordinate transforms:
- Camera -> Wall: p_w = R_wc^T @ (p_c - t_wc)
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

# If you want to optionally save the wall pose computed here:
OPTIONALLY_SAVE_WALL_POSE = False
WALL_POSE_SAVE_PATH = os.path.join(PROJECT_ROOT, "T_wall2cam_COMPUTED.npy")


TRIANGLE_EDGE_SIZE  = 0.15
RECTANGLE_LENGTH    = 0.85   # along wall x
RECTANGLE_HEIGHT    = 0.04   # along wall y (we'll draw 2*height tall band)
RECTANGLE_WIDTH     = 0.16   # along wall z (thickness)


def unproject_points(coords_2d, depth_map, intrinsics):
    """
    2D pixels -> 3D camera coordinates. coords_2d should be (M,2) in (x,y) with integer pixel indices.
    """
    fx, fy, cx, cy = intrinsics
    if not isinstance(coords_2d, np.ndarray):
        coords_2d = np.array(coords_2d)
    x_coords, y_coords = coords_2d[:, 0].astype(int), coords_2d[:, 1].astype(int)

    # Guard against out-of-bounds due to polygon edges touching borders
    H, W = depth_map.shape[:2]
    mask_in = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
    x_coords = x_coords[mask_in]
    y_coords = y_coords[mask_in]
    if x_coords.size == 0:
        return np.empty((0, 3), dtype=float)

    Z = depth_map[y_coords, x_coords]
    valid_mask = (Z > 0) & ~np.isnan(Z)
    x_coords, y_coords, Z = x_coords[valid_mask], y_coords[valid_mask], Z[valid_mask]
    if Z.size == 0:
        return np.empty((0, 3), dtype=float)

    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)


def calculate_local_frame(point_cloud):
    """
    PCA-based local frame from a point cloud.
    Returns: centroid, x_axis, y_axis, z_axis (all np.ndarray(3,))
    Guarantees a right-handed frame and consistent normal flipping (z may be flipped to your preference).
    """
    if point_cloud.shape[0] < 3:
        return None, None, None, None

    centroid = np.mean(point_cloud, axis=0)
    centered = point_cloud - centroid
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)  # eigenvectors are columns
    idx = np.argsort(vals)[::-1]
    v1, v2, v3 = vecs[:, idx[0]], vecs[:, idx[1]], vecs[:, idx[2]]

    # Enforce right-handed frame: z = x × y
    z_axis = np.cross(v1, v2)
    if np.linalg.norm(z_axis) < 1e-9:
        return None, None, None, None
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = v1 / np.linalg.norm(v1)
    # Recompute y to be orthonormal
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    z_axis = -z_axis

    return centroid, x_axis, y_axis, z_axis


# def calculate_local_frame(point_cloud):
#     """
#     PCA method to calculate the local coordinate frame for a point cloud.
#     """
#     if point_cloud.shape[0] < 3:
#         return None, None, None, None
#     centroid = np.mean(point_cloud, axis=0)
#     centered_points = point_cloud - centroid
#     covariance_matrix = np.cov(centered_points, rowvar=False)
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
#     sorted_indices = np.argsort(eigenvalues)[::-1]
#     x_axis = eigenvectors[:, sorted_indices[0]]
#     y_axis = eigenvectors[:, sorted_indices[1]]
#     z_axis = eigenvectors[:, sorted_indices[2]]
#     z_axis = -z_axis
#     return centroid, x_axis, y_axis, z_axis





def compute_pose_matrix(centroid, x_axis, y_axis, z_axis):
    """Build 4x4 pose from axes and centroid."""
    R = np.column_stack((x_axis/np.linalg.norm(x_axis),
                         y_axis/np.linalg.norm(y_axis),
                         z_axis/np.linalg.norm(z_axis)))
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3]  = centroid
    return T




if __name__ == "__main__":
    # Load data
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    brick_wall_vertices_2d  = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)
    num_frames, height, width = all_depth_maps.shape

    
    trowel_local_frames = []  # list of tuples: (centroid, x, y, z) in CAMERA coordinates
    for i in tqdm(range(num_frames), desc="Calculating Trowel Frames"):
        frame_vertices = trowel_vertices_2d_traj[i]
        if frame_vertices.size == 0:
            continue

        # rasterize polygon to mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [frame_vertices.astype(np.int32)], 1)
        rows, cols = np.where(mask == 1)
        if rows.size == 0:
            continue

        pixel_coords_2d = np.stack([cols, rows], axis=1)
        pts3d_cam = unproject_points(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
        if pts3d_cam.shape[0] < 3:
            continue

        local_frame = calculate_local_frame(pts3d_cam)
        if local_frame[0] is not None:
            trowel_local_frames.append(local_frame)

    
    
    wall_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(wall_mask, [brick_wall_vertices_2d.astype(np.int32)], 1)
    wr, wc = np.where(wall_mask == 1)
    wall_pixels = np.stack([wc, wr], axis=1)
    wall_pts3d_cam = unproject_points(wall_pixels, all_depth_maps[0], INTRINSICS)
    if wall_pts3d_cam.shape[0] < 3:
        raise RuntimeError("Wall point cloud too small. Check wall polygon or depth map.")

    centroid_w_c, x_w_c, y_w_c, z_w_c = calculate_local_frame(wall_pts3d_cam)
    if centroid_w_c is None:
        raise RuntimeError("Failed to compute wall local frame.")

    # Wall pose in CAMERA: ^cT_w
    T_wall2cam = compute_pose_matrix(centroid_w_c, x_w_c, y_w_c, z_w_c)
    R_wc = T_wall2cam[:3, :3]  # ^cR_w
    t_wc = T_wall2cam[:3, 3]   # ^ct_w

    if OPTIONALLY_SAVE_WALL_POSE:
        np.save(WALL_POSE_SAVE_PATH, T_wall2cam)

    # Convenience: camera->wall transform for points
    def cam_to_wall_points(Pc):
        """Pc: (N,3) camera points -> (N,3) wall-frame points"""
        if Pc.ndim == 1:
            Pc = Pc.reshape(1, 3)
        return (R_wc.T @ (Pc.T - t_wc.reshape(3, 1))).T


   
   
    trowel_centroids_cam = np.array([fr[0] for fr in trowel_local_frames])  # (N,3)
    trowel_centroids_wall = cam_to_wall_points(trowel_centroids_cam)        # (N,3)
    trowel_centroids_wall[:, 0] = -trowel_centroids_wall[:, 0]  # Flip x to match wall frame convention

    
    plotter = pv.Plotter(window_size=[1400, 900])
    plotter.set_background("white")

    # 1) Draw wall axes at origin of wall frame
    plotter.add_points(np.zeros((1, 3)), color="black", point_size=12, label="Wall Origin")
    arrow_scale = 0.1
    plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[-1, 0, 0]]),
                       mag=arrow_scale, color="red", label="Wall X-Axis")
    plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[0, 1, 0]]),
                       mag=arrow_scale, color="green", label="Wall Y-Axis")
    plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[0, 0, 1]]),
                       mag=arrow_scale, color="blue", label="Wall Z-Axis (Normal)")

    
    
    l, w, h = RECTANGLE_LENGTH / 2, RECTANGLE_WIDTH / 2, RECTANGLE_HEIGHT / 2
    # front face (z=0 plane band from y=-2h to y=0)
    v1f = np.array([-l, -2*h, 0])
    v2f = np.array([+l, -2*h, 0])
    v3f = np.array([+l,     0, 0])
    v4f = np.array([-l,     0, 0])
    # back face (z = +w)
    v1b, v2b, v3b, v4b = v1f.copy(), v2f.copy(), v3f.copy(), v4f.copy()
    v1b[2] += w; v2b[2] += w; v3b[2] += w; v4b[2] += w

    pts_w = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
    faces = np.hstack([
        [4, 0, 1, 2, 3],   # front
        [4, 4, 5, 6, 7],   # back
        [4, 0, 1, 5, 4],
        [4, 1, 2, 6, 5],
        [4, 2, 3, 7, 6],
        [4, 3, 0, 4, 7],
    ]).astype(np.int64)
    wall_mesh = pv.PolyData(pts_w, faces)
    plotter.add_mesh(
        wall_mesh, color="#D95319", opacity=1.0,
        show_edges=True, edge_color="black", line_width=2,
        label="Wall Volume (wall frame)"
    )

    
    
    plotter.add_mesh(pv.Spline(trowel_centroids_wall, 1000),
                     color="blue", line_width=5, label="Trowel Centroid Path")
    plotter.add_points(trowel_centroids_wall[0],  color="green", point_size=15,
                       render_points_as_spheres=True, label="Start")
    plotter.add_points(trowel_centroids_wall[-1], color="red",   point_size=15,
                       render_points_as_spheres=True, label="End")

    
    
    DRAW_TROWEL_TRIANGLES = True
    if DRAW_TROWEL_TRIANGLES:
        for i, (cent_c, x_c, y_c, z_c) in enumerate(trowel_local_frames):
            # Tri in the trowel local x-y plane, centered at centroid
            length, width_tri = TRIANGLE_EDGE_SIZE, TRIANGLE_EDGE_SIZE / 2
            v1_2d = np.array([-length/2, 0])
            v2_2d = np.array([+length/2, -width_tri / 2])
            v3_2d = np.array([+length/2, +width_tri / 2])

            # 3D in CAMERA first
            v1_c = cent_c + v1_2d[0] * x_c + v1_2d[1] * y_c
            v2_c = cent_c + v2_2d[0] * x_c + v2_2d[1] * y_c
            v3_c = cent_c + v3_2d[0] * x_c + v3_2d[1] * y_c

            # Then transform to WALL
            tri_w = cam_to_wall_points(np.stack([v1_c, v2_c, v3_c], axis=0))
            tri_w[:, 0] *= -1
            face = np.hstack([3, 0, 1, 2])
            tri_mesh = pv.PolyData(tri_w, face)
            plotter.add_mesh(tri_mesh, color="grey", opacity=0.4, show_edges=True)

    plotter.add_legend()
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    plotter.show()
