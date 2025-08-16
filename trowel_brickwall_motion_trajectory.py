"""
Static visualization & relative-pose visualization.

What it shows:
1) In CAMERA frame:
   - Brick wall side-surface point cloud and best-fit plane + axes
   - Trowel centroid trajectory and per-frame oriented triangle glyph

2) In BRICK frame (tool->target, camera-free):
   - Brick wall cloud at z≈0 and its plane rectangle at origin
   - Trowel centroid path and per-frame oriented triangle glyph
   - Saves T^{brick}_{trowel}[i] as (N,4,4) to 'trowel_pose_in_brick_frame.npy'
"""

import os
import sys
import numpy as np
import pyvista as pv
import cv2
from tqdm import tqdm

# -------------------- Configuration --------------------
# Camera intrinsics
FX = 836.0
FY = 836.0
CX = 979.0
CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY], dtype=float)

# File paths (assuming the script is run from the project's root directory)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
TROWEL_TIP_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_tip_polygon_vertices.npy")
BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# Visualization Parameters
TRIANGLE_EDGE_SIZE = 0.15  # meters (glyph footprint for trowel)
RECTANGLE_LENGTH = 0.85    # meters (brick plane visualization)
RECTANGLE_WIDTH  = 0.10    # meters

# Optional point subsampling for speed (set to None to disable)
SUBSAMPLE_BRICK  = 40000
SUBSAMPLE_TROWEL = 8000

# -------------------- Helpers --------------------
def _clip_polygon_to_image(poly_xy, W, H):
    """Clip (N,2) polygon to image bounds and cast to int32 for cv2."""
    if poly_xy is None or len(poly_xy) == 0:
        return None
    poly = np.asarray(poly_xy, dtype=float).copy()
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    return poly.astype(np.int32)

def polygon_to_point_cloud(poly_xy, depth_map, intrinsics):
    """Fill polygon -> mask -> unproject to 3D."""
    H, W = depth_map.shape[:2]
    poly = _clip_polygon_to_image(poly_xy, W, H)
    if poly is None or poly.size == 0:
        return np.empty((0, 3), dtype=float)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    rows, cols = np.where(mask == 1)
    if rows.size == 0:
        return np.empty((0, 3), dtype=float)
    pix2d = np.stack([cols, rows], axis=-1)
    return unproject_points(pix2d, depth_map, intrinsics)

def unproject_points(coords_2d, depth_map, intrinsics):
    """Projects a list of 2D pixel coordinates into 3D (camera frame)."""
    fx, fy, cx, cy = intrinsics
    if not isinstance(coords_2d, np.ndarray):
        coords_2d = np.array(coords_2d)
    x_coords = coords_2d[:, 0].astype(np.int64)
    y_coords = coords_2d[:, 1].astype(np.int64)

    # Safe bounds just in case
    H, W = depth_map.shape[:2]
    valid_xy = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
    x_coords = x_coords[valid_xy]
    y_coords = y_coords[valid_xy]
    if x_coords.size == 0:
        return np.empty((0, 3), dtype=float)

    depths = depth_map[y_coords, x_coords]
    valid_mask = (depths > 0) & np.isfinite(depths)
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=float)

    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    Z = depths[valid_mask].astype(float)

    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).astype(float)

def calculate_local_frame(point_cloud):
    """
    PCA method to calculate a local frame for a (roughly planar) point cloud.
    Returns centroid, x_axis, y_axis, z_axis (not guaranteed right-handed).
    """
    if point_cloud is None or point_cloud.shape[0] < 3:
        return None, None, None, None
    centroid = np.mean(point_cloud, axis=0)
    Xc = point_cloud - centroid
    # Use SVD for stability
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T  # columns are principal directions
    # For a plane: smallest variance is normal (last singular value)
    # Largest two span the plane
    x_axis = V[:, 0]
    y_axis = V[:, 1]
    z_axis = V[:, 2]
    # By convention, you may choose to flip z to face outward if needed; we’ll fix later
    return centroid, x_axis, y_axis, z_axis

def _normalize(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def pose_from_axes(centroid, x_axis, y_axis, z_axis, prev_R=None):
    """
    Build a clean, right-handed, temporally-consistent pose (R, p) from raw PCA axes.
    R columns are local x,y,z expressed in camera basis.
    """
    if centroid is None:
        return None, None

    x = _normalize(x_axis)
    z = _normalize(z_axis)
    # Rebuild orthonormal triad with Gram-Schmidt & right-handedness
    y = _normalize(np.cross(z, x))
    x = _normalize(np.cross(y, z))
    z = _normalize(z)

    R = np.column_stack([x, y, z])

    # Enforce right-handedness
    if np.linalg.det(R) < 0:
        # Flip z (or any one axis) to fix the determinant
        R[:, 2] = -R[:, 2]

    # Temporal sign consistency (avoid flips over time)
    if prev_R is not None:
        for j in range(3):
            if np.dot(R[:, j], prev_R[:, j]) < 0:
                R[:, j] = -R[:, j]

    return R, centroid.astype(float)

def invert_Rp(R, p):
    Rt = R.T
    return Rt, -Rt @ p

def compose_Rp(Ra, pa, Rb, pb):
    """[Ra pa;0 1] * [Rb pb;0 1] -> (Ra Rb, Ra pb + pa)"""
    return Ra @ Rb, Ra @ pb + pa

def subsample_points(pts, max_n):
    if pts is None or pts.shape[0] == 0 or max_n is None:
        return pts
    if pts.shape[0] <= max_n:
        return pts
    idx = np.random.choice(pts.shape[0], size=max_n, replace=False)
    return pts[idx]

# -------------------- Main --------------------
if __name__ == "__main__":
    np.random.seed(0)

    print("Loading input data...")
    all_depth_maps = np.load(DEPTH_MAP_PATH)  # (N,H,W)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    # tip polys optional; not used here but loaded to match your structure
    _trowel_tip_vertices_2d_traj = np.load(TROWEL_TIP_VERTICES_2D_PATH, allow_pickle=True)
    brick_wall_vertices_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)

    if all_depth_maps.ndim != 3:
        raise ValueError("Depth map array must be (N, H, W).")
    num_frames, height, width = all_depth_maps.shape
    print(f"Depth maps: {all_depth_maps.shape}")
    print(f"Trowel poly traj dtype={type(trowel_vertices_2d_traj)} len={len(trowel_vertices_2d_traj)}")
    print(f"Brick wall poly: shape {brick_wall_vertices_2d.shape}")

    # ------------- Build per-frame Trowel local frames (from polygons) -------------
    trowel_local_frames = []  # list of (centroid, x_axis, y_axis, z_axis)
    valid_frame_idx = []      # indices that produced a valid frame

    print("Calculating Trowel Frames (PCA from polygon region points)...")
    for i in tqdm(range(num_frames), desc="Trowel Frames"):
        frame_vertices = trowel_vertices_2d_traj[i]
        if frame_vertices is None or len(frame_vertices) == 0:
            continue

        # Make 3D cloud from polygon filled pixels
        pc3d = polygon_to_point_cloud(frame_vertices, all_depth_maps[i], INTRINSICS)
        if pc3d.size == 0:
            continue

        # Optional subsample for speed
        pc3d = subsample_points(pc3d, SUBSAMPLE_TROWEL)

        # PCA local frame
        centroid, x_axis, y_axis, z_axis = calculate_local_frame(pc3d)
        if centroid is None:
            continue
        trowel_local_frames.append((centroid, x_axis, y_axis, z_axis))
        valid_frame_idx.append(i)

    if len(trowel_local_frames) == 0:
        raise RuntimeError("No valid trowel frames were computed.")

    # ------------- Project & analyze the static brick wall -------------
    print("Projecting and analyzing the static brick wall...")
    brick_cloud_cam = polygon_to_point_cloud(brick_wall_vertices_2d, all_depth_maps[0], INTRINSICS)
    brick_cloud_cam = subsample_points(brick_cloud_cam, SUBSAMPLE_BRICK)

    if brick_cloud_cam.size == 0:
        raise RuntimeError("Brick wall point cloud is empty after projection.")

    wall_centroid, wall_x, wall_y, wall_z = calculate_local_frame(brick_cloud_cam)
    if wall_centroid is None:
        raise RuntimeError("Failed to compute brick wall local frame from its point cloud.")

    # ------------- Build clean poses in CAMERA frame -------------
    # Brick pose in camera (once)
    R_cam_brick, p_cam_brick = pose_from_axes(wall_centroid, wall_x, wall_y, wall_z, prev_R=None)

    # Per-frame trowel poses in camera
    R_cam_trowel_list, p_cam_trowel_list = [], []
    prev_R_t = None
    for (centroid, x_axis, y_axis, z_axis) in trowel_local_frames:
        R_cam_t, p_cam_t = pose_from_axes(centroid, x_axis, y_axis, z_axis, prev_R=prev_R_t)
        R_cam_trowel_list.append(R_cam_t)
        p_cam_trowel_list.append(p_cam_t)
        prev_R_t = R_cam_t.copy()

    # ------------- Convert to BRICK frame: T^{brick}_{trowel} -------------
    R_brick_cam, p_brick_cam = invert_Rp(R_cam_brick, p_cam_brick)

    R_brick_trowel_list, p_brick_trowel_list, T_brick_trowel_list = [], [], []
    for R_cam_t, p_cam_t in zip(R_cam_trowel_list, p_cam_trowel_list):
        R_bt = R_brick_cam @ R_cam_t
        p_bt = R_brick_cam @ (p_cam_t - p_cam_brick)
        R_brick_trowel_list.append(R_bt)
        p_brick_trowel_list.append(p_bt)

        T_bt = np.eye(4, dtype=float)
        T_bt[:3, :3] = R_bt
        T_bt[:3,  3] = p_bt
        T_brick_trowel_list.append(T_bt)

    T_brick_trowel = np.stack(T_brick_trowel_list, axis=0)  # (N_valid, 4, 4)
    out_path = os.path.join(PROJECT_ROOT, "trowel_pose_in_brick_frame.npy")
    np.save(out_path, T_brick_trowel)
    print(f"Saved relative poses to: {out_path}  (shape {T_brick_trowel.shape})")
    print(f"Valid frames used: {len(valid_frame_idx)} / {num_frames}")

    # -------------------- Visualization (CAMERA frame) --------------------
    print("\nSetting up CAMERA-FRAME plot...")
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # Brick cloud
    plotter.add_points(brick_cloud_cam, style='points', color='#D95319',
                       render_points_as_spheres=True, point_size=3, label='Brick Wall Cloud (cam)')

    # Brick frame axes & best-fit rectangle (in CAMERA frame)
    arrow_scale = 0.12
    plotter.add_points(wall_centroid, color='black', point_size=10, label='Wall Centroid (cam)')
    plotter.add_arrows(cent=np.array([wall_centroid]), direction=np.array([R_cam_brick[:, 0]]),
                       mag=arrow_scale, color='red',   label='Wall X (cam)')
    plotter.add_arrows(cent=np.array([wall_centroid]), direction=np.array([R_cam_brick[:, 1]]),
                       mag=arrow_scale, color='green', label='Wall Y (cam)')
    plotter.add_arrows(cent=np.array([wall_centroid]), direction=np.array([R_cam_brick[:, 2]]),
                       mag=arrow_scale, color='blue',  label='Wall Z (cam)')

    # Brick plane rectangle (cam frame)
    l, w = RECTANGLE_LENGTH / 2, RECTANGLE_WIDTH / 2
    rect_2d = np.array([[-l, -w], [ l, -w], [ l,  w], [-l,  w]], dtype=float)
    rect_cam = (wall_centroid
                + rect_2d[0,0]*R_cam_brick[:,0] + rect_2d[0,1]*R_cam_brick[:,1],
                wall_centroid
                + rect_2d[1,0]*R_cam_brick[:,0] + rect_2d[1,1]*R_cam_brick[:,1],
                wall_centroid
                + rect_2d[2,0]*R_cam_brick[:,0] + rect_2d[2,1]*R_cam_brick[:,1],
                wall_centroid
                + rect_2d[3,0]*R_cam_brick[:,0] + rect_2d[3,1]*R_cam_brick[:,1])
    rect_cam = np.vstack(rect_cam)
    face = np.hstack([4, 0, 1, 2, 3]).astype(np.int64)
    plotter.add_mesh(pv.PolyData(rect_cam, face), color='lightgreen', opacity=0.8,
                     show_edges=True, edge_color='black', line_width=3, label='Wall Plane (cam)')

    # Trowel centroid path (cam frame)
    p_traj_cam = np.stack(p_cam_trowel_list, axis=0)
    plotter.add_mesh(pv.Spline(p_traj_cam, 1000), color="blue", line_width=5, label="Trowel Path (cam)")
    plotter.add_points(p_traj_cam[0],  color='green', point_size=15, render_points_as_spheres=True, label='Start (cam)')
    plotter.add_points(p_traj_cam[-1], color='red',   point_size=15, render_points_as_spheres=True, label='End (cam)')

    # Per-frame trowel oriented triangle (cam frame)
    for i, (R_ct, p_ct) in enumerate(zip(R_cam_trowel_list, p_cam_trowel_list)):
        x_ct = R_ct[:, 0]
        y_ct = R_ct[:, 1]
        length, width_t = TRIANGLE_EDGE_SIZE, TRIANGLE_EDGE_SIZE / 2
        v1_2d = np.array([-length/2, 0.0])
        v2_2d = np.array([ length/2, -width_t/2])
        v3_2d = np.array([ length/2,  width_t/2])
        v1 = p_ct + v1_2d[0]*x_ct + v1_2d[1]*y_ct
        v2 = p_ct + v2_2d[0]*x_ct + v2_2d[1]*y_ct
        v3 = p_ct + v3_2d[0]*x_ct + v3_2d[1]*y_ct
        tri_pts = np.vstack([v1, v2, v3])
        face3 = np.hstack([3, 0, 1, 2]).astype(np.int64)
        opacity = (i + 1) / len(R_cam_trowel_list)
        plotter.add_mesh(pv.PolyData(tri_pts, face3), color='purple', opacity=opacity, show_edges=True)

    plotter.add_legend()
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)

    print("\nShowing CAMERA-FRAME plot... (close to open brick-frame view)")
    plotter.show()

    # -------------------- Visualization (BRICK frame) --------------------
    print("\nSetting up BRICK-FRAME plot...")
    plotter2 = pv.Plotter(window_size=[1200, 800])
    plotter2.set_background('white')

    # Brick cloud in BRICK frame (should lie on z≈0)
    brick_cloud_brick = (R_brick_cam @ (brick_cloud_cam - p_cam_brick).T).T
    plotter2.add_points(brick_cloud_brick, style='points', color='#D95319',
                        render_points_as_spheres=True, point_size=3, label='Brick Wall Cloud (brick)')

    # Brick-plane rectangle at origin on z=0
    rect_brick = np.array([
        [-l, -w, 0.0],
        [ l, -w, 0.0],
        [ l,  w, 0.0],
        [-l,  w, 0.0],
    ], dtype=float)
    plotter2.add_mesh(pv.PolyData(rect_brick, face),
                      color='lightgreen', opacity=0.8, show_edges=True,
                      edge_color='black', line_width=3, label='Wall Plane (brick)')

    # Brick frame axes at origin
    origin = np.array([[0.0, 0.0, 0.0]])
    plotter2.add_arrows(cent=origin, direction=np.array([[1,0,0]]), mag=arrow_scale, color='red',   label='Brick X')
    plotter2.add_arrows(cent=origin, direction=np.array([[0,1,0]]), mag=arrow_scale, color='green', label='Brick Y')
    plotter2.add_arrows(cent=origin, direction=np.array([[0,0,1]]), mag=arrow_scale, color='blue',  label='Brick Z')

    # Trowel centroid path in BRICK frame
    p_traj_brick = np.stack(p_brick_trowel_list, axis=0)
    plotter2.add_mesh(pv.Spline(p_traj_brick, 1000), color="blue", line_width=5, label="Trowel Path (brick)")
    plotter2.add_points(p_traj_brick[0],  color='green', point_size=15, render_points_as_spheres=True, label='Start (brick)')
    plotter2.add_points(p_traj_brick[-1], color='red',   point_size=15, render_points_as_spheres=True, label='End (brick)')

    # Per-frame trowel oriented triangle in BRICK frame
    for i, (R_bt, p_bt) in enumerate(zip(R_brick_trowel_list, p_brick_trowel_list)):
        x_bt = R_bt[:, 0]
        y_bt = R_bt[:, 1]
        length, width_t = TRIANGLE_EDGE_SIZE, TRIANGLE_EDGE_SIZE / 2
        v1_2d = np.array([-length/2, 0.0])
        v2_2d = np.array([ length/2, -width_t/2])
        v3_2d = np.array([ length/2,  width_t/2])
        v1 = p_bt + v1_2d[0]*x_bt + v1_2d[1]*y_bt
        v2 = p_bt + v2_2d[0]*x_bt + v2_2d[1]*y_bt
        v3 = p_bt + v3_2d[0]*x_bt + v3_2d[1]*y_bt
        tri_pts = np.vstack([v1, v2, v3])
        face3 = np.hstack([3, 0, 1, 2]).astype(np.int64)
        opacity = (i + 1) / len(R_brick_trowel_list)
        plotter2.add_mesh(pv.PolyData(tri_pts, face3), color='purple', opacity=opacity, show_edges=True)

    plotter2.add_legend()
    plotter2.camera.azimuth = -60
    plotter2.camera.elevation = 25
    plotter2.camera.zoom(1.3)

    print("\nShowing BRICK-FRAME plot... (wall should lie on z≈0)")
    plotter2.show()

    print("\nDone.")
