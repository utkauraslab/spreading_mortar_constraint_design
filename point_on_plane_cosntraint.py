
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Fit point-on-plane contact for the trowel tip during spreading.

# Inputs (camera frame):
# - depth_map_cross_frames_refined.npy  : (N, H, W) depth maps (meters)
# - trowel_polygon_vertices.npy         : list/array of per-frame polygon vertices in image pixels (x,y)
# - brick_wall_side_surface.npy         : single polygon vertices (x,y) on the brick top surface (one frame is fine)

# Outputs:
# - plane_params_wall_cam.npy : {'n': (3,), 'd': float}      with plane n^T x - d = 0
# - tip_points_cam.npy        : (M,3) tip points used (M <= N)
# - residuals.npy             : (M,) signed distances along n (meters)
# - tilt_degrees.npy          : (M,) angle between trowel surface normal and wall normal (deg)
# - frames_used.npy           : (M,) indices of frames used
# - stats.txt                 : human-readable residual/tilt stats

# Usage:
# - Just run. Optionally edit USER OPTIONS below (TIP_MODE, METHOD, etc).
# """

# import os
# import sys
# import json
# import numpy as np
# import cv2

# # =======================
# # USER OPTIONS
# # =======================
# TIP_MODE = "canonical"     # 'extreme' (recommended) or 'canonical'
# METHOD   = "huber"       # 'ls' | 'median' | 'huber'
# HUBER_DELTA = 1e-3       # meters; ~1 mm transition for Huber
# SAVE_DIR = "."           # where to save outputs

# # Canonical triangle geometry (only used if TIP_MODE='canonical')
# TRIANGLE_EDGE_SIZE = 0.15   # meters; 'length' of the canonical triangle along local +x

# # Brick top “box” for viz (optional dimensions, not used in fitting)
# RECTANGLE_LENGTH   = 0.85   # along x_w (meters)
# RECTANGLE_HEIGHT   = 0.04   # along y_w (meters)
# RECTANGLE_WIDTH    = 0.16   # offset along z_w (meters) for thickened viz

# # Camera intrinsics (fx, fy, cx, cy)
# FX = 836.0
# FY = 836.0
# CX = 979.0
# CY = 632.0
# INTRINSICS = np.array([FX, FY, CX, CY])

# # Project paths (edit if you placed files elsewhere)
# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
# TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
# BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# # =======================
# # Utils
# # =======================
# def unproject_points(coords_2d, depth_map, intrinsics):
#     """Unproject (x,y) pixels to camera 3D using per-pixel depth (meters)."""
#     fx, fy, cx, cy = intrinsics
#     coords_2d = np.asarray(coords_2d, dtype=np.int32)
#     x_pix, y_pix = coords_2d[:, 0], coords_2d[:, 1]
#     Z = depth_map[y_pix, x_pix]
#     valid = np.isfinite(Z) & (Z > 0)
#     x_pix, y_pix, Z = x_pix[valid], y_pix[valid], Z[valid]
#     X = (x_pix - cx) * Z / fx
#     Y = (y_pix - cy) * Z / fy
#     return np.stack([X, Y, Z], axis=-1)

# def pca_frame(points):
#     """
#     PCA frame (camera coords) from a 3D point cloud:
#     - returns centroid and (x,y,z) as the principal axes (sorted by variance).
#     - then orthonormalizes and enforces right-handedness.
#     """
#     P = np.asarray(points)
#     if P.shape[0] < 3:
#         return None, None, None, None
#     C = P.mean(axis=0)
#     Q = P - C
#     # Covariance & eigen-decomposition
#     cov = np.cov(Q, rowvar=False)
#     w, V = np.linalg.eigh(cov)  # ascending eigenvalues
#     order = np.argsort(w)[::-1] # descending
#     x = V[:, order[0]]
#     y = V[:, order[1]]
#     z = V[:, order[2]]          # smallest variance ⇒ normal-like

#     # Orthonormalize and make right-handed (Gram-Schmidt + det>0)
#     x = x / (np.linalg.norm(x) + 1e-12)
#     y = y - (x @ y) * x
#     y = y / (np.linalg.norm(y) + 1e-12)
#     z = np.cross(x, y)
#     nz = np.linalg.norm(z)
#     if nz < 1e-12:
#         # fallback via SVD if nearly collinear
#         R0 = np.column_stack([x, y, V[:, order[2]]])
#         U, _, Vt = np.linalg.svd(R0, full_matrices=False)
#         R = U @ Vt
#     else:
#         z = z / nz
#         R = np.column_stack([x, y, z])

#     if np.linalg.det(R) < 0:
#         # Flip y to enforce right-handedness
#         y = -y
#         z = np.cross(x, y)
#         z = z / (np.linalg.norm(z) + 1e-12)
#         R = np.column_stack([x, y, z])

#     return C, R[:, 0], R[:, 1], R[:, 2]  # centroid, x, y, z

# def np2py(obj):
#     """Make NumPy types JSON‑serializable."""
#     import numpy as np
#     if isinstance(obj, np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (np.floating, np.integer, np.bool_)):
#         return obj.item()
#     return obj


# def huber_weights(r, delta):
#     a = np.abs(r)
#     w = np.ones_like(r)
#     mask = a > delta
#     w[mask] = delta / (a[mask] + 1e-12)
#     return w

# def fit_offset_known_normal(points_cam, n_cam, method="ls", huber_delta=1e-3):
#     """
#     Fit offset d in plane n^T x - d = 0, given unit normal n and points x.
#     Returns d, residuals (n^T x - d), slack (max r^2), stats dict.
#     """
#     n = n_cam / (np.linalg.norm(n_cam) + 1e-12)
#     proj = points_cam @ n  # shape (N,)

#     if method == "ls":
#         d = float(np.mean(proj))
#     elif method == "median":
#         d = float(np.median(proj))
#     elif method == "huber":
#         d = float(np.mean(proj))
#         for _ in range(20):
#             r = proj - d
#             w = huber_weights(r, huber_delta)
#             d_new = float(np.sum(w * proj) / (np.sum(w) + 1e-12))
#             if abs(d_new - d) < 1e-10:
#                 break
#             d = d_new
#     else:
#         raise ValueError("method must be 'ls', 'median', or 'huber'")

#     residuals = proj - d
#     stats = {
#         "d_meters": d,
#         "mean_abs_mm": 1000.0 * float(np.mean(np.abs(residuals))),
#         "p98_abs_mm": 1000.0 * float(np.percentile(np.abs(residuals), 98)),
#         "max_abs_mm": 1000.0 * float(np.max(np.abs(residuals))),
#         "std_mm":     1000.0 * float(np.std(residuals)),
#         "num_points": int(points_cam.shape[0]),
#     }
#     slack = float(np.max(residuals ** 2))  # paper-style (max r^2)
#     return d, residuals, slack, stats

# def angle_between_unit_vectors(u, v):
#     """Angle in radians between two (approximately unit) vectors."""
#     uu = u / (np.linalg.norm(u) + 1e-12)
#     vv = v / (np.linalg.norm(v) + 1e-12)
#     c = np.clip(uu @ vv, -1.0, 1.0)
#     return float(np.arccos(c))


# # =======================
# # Main
# # =======================
# def main():
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # ---- Load data ----
#     all_depth_maps = np.load(DEPTH_MAP_PATH)                 # (N,H,W)
#     trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
#     brick_wall_vertices_2d   = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)
#     N, H, W = all_depth_maps.shape

#     # ---- Brick top surface: get normal (camera frame) via PCA ----
#     # Build mask from provided polygon (assumed top surface)
#     wall_mask = np.zeros((H, W), dtype=np.uint8)
#     cv2.fillPoly(wall_mask, [brick_wall_vertices_2d], 1)
#     rows, cols = np.where(wall_mask == 1)
#     wall_pix = np.vstack([cols, rows]).T
#     wall_cloud = unproject_points(wall_pix, all_depth_maps[0], INTRINSICS)

#     wall_C, wall_x, wall_y, wall_z = pca_frame(wall_cloud)
#     if wall_C is None:
#         raise RuntimeError("Brick wall point cloud too small to estimate a plane.")

#     n_wall = wall_z / (np.linalg.norm(wall_z) + 1e-12)  # unit normal (camera frame)

#     # ---- Extract trowel tip points per frame (camera frame) ----
#     tip_points = []
#     frames_used = []
#     tilt_deg = []

#     for i in range(N):
#         verts2d = trowel_vertices_2d_traj[i]
#         if verts2d.size == 0:
#             continue

#         # Build 3D cloud of the trowel polygon in frame i
#         m = np.zeros((H, W), dtype=np.uint8)
#         cv2.fillPoly(m, [verts2d], 1)
#         rr, cc = np.where(m == 1)
#         pix = np.vstack([cc, rr]).T
#         cloud = unproject_points(pix, all_depth_maps[i], INTRINSICS)
#         if cloud.shape[0] < 10:
#             continue

#         # Local PCA frame for the trowel in this frame
#         C_t, x_t, y_t, z_t = pca_frame(cloud)
#         if C_t is None:
#             continue

#         # Choose tip
#         if TIP_MODE.lower() == "extreme":
#             # Find extreme point along -x_t (local), i.e., min projection on x_t
#             rel = cloud - C_t
#             xproj = rel @ (x_t / (np.linalg.norm(x_t) + 1e-12))
#             idx = int(np.argmin(xproj))
#             p_tip = cloud[idx]
#         else:
#             # Canonical tip at (-L/2, 0, 0) in local; transform to camera
#             tip_local = np.array([-TRIANGLE_EDGE_SIZE / 2.0, 0.0, 0.0])
#             R_t = np.column_stack([x_t, y_t, z_t])  # 3x3
#             p_tip = C_t + R_t @ tip_local

#         tip_points.append(p_tip)
#         frames_used.append(i)

#         # Optional: tilt of trowel surface vs wall surface (angle between normals)
#         # We treat z_t (smallest variance axis) as the trowel surface normal.
#         theta = angle_between_unit_vectors(z_t, n_wall)
#         tilt_deg.append(np.degrees(theta))

#     tip_points = np.asarray(tip_points)
#     frames_used = np.asarray(frames_used, dtype=np.int32)
#     tilt_deg = np.asarray(tilt_deg)

#     if tip_points.shape[0] < 3:
#         raise RuntimeError("Not enough valid frames with trowel to fit contact plane.")

#     # ---- Fit offset d for plane n^T x - d = 0 ----
#     d, residuals, slack, stats = fit_offset_known_normal(
#         tip_points, n_wall, method=METHOD, huber_delta=HUBER_DELTA
#     )

#     # ---- Save outputs ----
#     plane_pack = {"n": n_wall.astype(float), "d": float(d)}
#     # np.save(os.path.join(SAVE_DIR, "plane_params_wall_cam.npy"), plane_pack)
#     # np.save(os.path.join(SAVE_DIR, "tip_points_cam.npy"), tip_points)
#     # np.save(os.path.join(SAVE_DIR, "residuals.npy"), residuals)
#     # np.save(os.path.join(SAVE_DIR, "tilt_degrees.npy"), tilt_deg)
#     # np.save(os.path.join(SAVE_DIR, "frames_used.npy"), frames_used)

#     # with open(os.path.join(SAVE_DIR, "stats.txt"), "w") as f:
#     #     f.write(json.dumps({
#     #         "fit_method": METHOD,
#     #         "huber_delta_m": HUBER_DELTA,
#     #         "tip_mode": TIP_MODE,
#     #         "plane": plane_pack,
#     #         "residual_stats": stats,
#     #         "tilt_stats_deg": {
#     #             "mean": float(np.mean(tilt_deg)) if tilt_deg.size else None,
#     #             "p98":  float(np.percentile(tilt_deg, 98)) if tilt_deg.size else None,
#     #             "max":  float(np.max(tilt_deg)) if tilt_deg.size else None,
#     #             "std":  float(np.std(tilt_deg)) if tilt_deg.size else None,
#     #         },
#     #         "num_frames_used": int(tip_points.shape[0])
#     #     }, indent=2))

#     summary = {
#         "fit_method": METHOD,
#         "huber_delta_m": float(HUBER_DELTA),
#         "tip_mode": TIP_MODE,
#         "plane": {"n": np2py(n_wall), "d": float(d)},
#         "residual_stats": {
#             "mean_abs_mm": float(stats["mean_abs_mm"]),
#             "p98_abs_mm":  float(stats["p98_abs_mm"]),
#             "max_abs_mm":  float(stats["max_abs_mm"]),
#             "std_mm":      float(stats["std_mm"]),
#             "num_points":  int(stats["num_points"]),
#         },
#         "tilt_stats_deg": {
#             "mean": float(np.mean(tilt_deg)) if tilt_deg.size else None,
#             "p98":  float(np.percentile(tilt_deg, 98)) if tilt_deg.size else None,
#             "max":  float(np.max(tilt_deg)) if tilt_deg.size else None,
#             "std":  float(np.std(tilt_deg)) if tilt_deg.size else None,
#         },
#         "num_frames_used": int(tip_points.shape[0]),
#         "frames_used": np2py(frames_used),
#     }

#     with open(os.path.join(SAVE_DIR, "stats.txt"), "w") as f:
#         f.write(json.dumps(summary, default=np2py, indent=2))


#     # ---- Console summary ----
#     print("\n=== Point-on-plane fit (camera frame) ===")
#     print(f"n (unit): {n_wall}")
#     print(f"d (meters): {d:.6f}   -> plane:  n^T x - d = 0")
#     print(f"Residuals: mean|r|={stats['mean_abs_mm']:.2f} mm, p98|r|={stats['p98_abs_mm']:.2f} mm, "
#           f"max|r|={stats['max_abs_mm']:.2f} mm, std={stats['std_mm']:.2f} mm, N={stats['num_points']}")
#     if tilt_deg.size:
#         print(f"Tilt (deg): mean={np.mean(tilt_deg):.2f}, p98={np.percentile(tilt_deg,98):.2f}, "
#               f"max={np.max(tilt_deg):.2f}, std={np.std(tilt_deg):.2f}")
#     print(f"Slack (max r^2): {slack:.6e}")
    

# if __name__ == "__main__":
#     main()

















"""
Estimate the brick TOP surface plane and fit a point-on-plane constraint
for the trowel tip trajectory.

Inputs (camera frame):
- depth_map_cross_frames_refined.npy  : (N, H, W) depth maps (meters)
- trowel_polygon_vertices.npy         : list/array of per-frame polygon vertices in image pixels (x,y)
- brick_wall_side_surface.npy         : SINGLE polygon (x,y) of the *side surface* (same frame as depth[0])

Outputs:
- plane_top_cam.npy      : {'n': (3,), 'd': float}    plane: n^T x - d = 0
- tip_points_cam.npy     : (M,3) tip points used (M <= N)
- residuals.npy          : (M,) signed distances (meters)
- tilt_degrees.npy       : (M,) angle between trowel surface normal and TOP normal (deg)
- frames_used.npy        : (M,) indices of frames used
- stats.txt              : JSON summary (plain-text)

Usage:
- python3 fit_top_plane_point_on_plane.py
"""

import os
import json
import numpy as np
import cv2

# =======================
# USER OPTIONS
# =======================
TIP_MODE      = "canonical"    # 'extreme' (recommended) or 'canonical'
FIT_METHOD    = "huber"      # 'ls' | 'median' | 'huber'
HUBER_DELTA   = 1e-3         # meters (~1 mm) for Huber
SAVE_DIR      = "."          # output folder

# For canonical trowel tip (only if TIP_MODE='canonical')
TRIANGLE_EDGE_SIZE = 0.15    # meters (tip at (-L/2, 0, 0) in local)

# Camera intrinsics (fx, fy, cx, cy)
FX, FY, CX, CY = 836.0, 836.0, 979.0, 632.0
INTRINSICS = np.array([FX, FY, CX, CY], dtype=np.float64)

# Input paths
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
SIDE_SURFACE_POLY_PATH  = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# =======================
# Helpers
# =======================
def np2py(obj):
    """Make NumPy types JSON-serializable."""
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
        return obj.item()
    return obj

def unproject_points(coords_2d, depth_map, intrinsics):
    """Unproject (x,y) pixels to camera 3D using per-pixel depth (meters)."""
    fx, fy, cx, cy = intrinsics
    coords_2d = np.asarray(coords_2d, dtype=np.int32)
    # clamp to image bounds for safety
    h, w = depth_map.shape[:2]
    x_pix = np.clip(coords_2d[:, 0], 0, w - 1)
    y_pix = np.clip(coords_2d[:, 1], 0, h - 1)
    Z = depth_map[y_pix, x_pix]
    ok = np.isfinite(Z) & (Z > 0)
    x_pix, y_pix, Z = x_pix[ok], y_pix[ok], Z[ok]
    X = (x_pix - cx) * Z / fx
    Y = (y_pix - cy) * Z / fy
    return np.stack([X, Y, Z], axis=-1)

def pca_frame(points):
    """
    PCA frame (camera coords) from a 3D point cloud:
      returns centroid and (x,y,z) principal axes (orthonormal, right-handed).
    """
    P = np.asarray(points)
    if P.shape[0] < 3:
        return None, None, None, None
    C = P.mean(axis=0)
    Q = P - C
    cov = np.cov(Q, rowvar=False)
    w, V = np.linalg.eigh(cov)          # ascending eigenvalues
    order = np.argsort(w)[::-1]         # descending
    x = V[:, order[0]]
    y = V[:, order[1]]
    # third axis via cross to enforce orthonormality/right-handedness
    x = x / (np.linalg.norm(x) + 1e-12)
    y = y - (x @ y) * x
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    nz = np.linalg.norm(z)
    if nz < 1e-12:
        # fallback via SVD if degenerate
        R0 = np.column_stack([x, y, V[:, order[2]]])
        U, _, Vt = np.linalg.svd(R0, full_matrices=False)
        R = U @ Vt
    else:
        z = z / nz
        R = np.column_stack([x, y, z])

    if np.linalg.det(R) < 0:
        y = -y
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-12)
        R = np.column_stack([x, y, z])
    return C, R[:, 0], R[:, 1], R[:, 2]

def huber_weights(r, delta):
    a = np.abs(r)
    w = np.ones_like(r)
    mask = a > delta
    w[mask] = delta / (a[mask] + 1e-12)
    return w

def fit_offset_known_normal(points_cam, n_cam, method="ls", huber_delta=1e-3):
    """
    Fit offset d in plane n^T x - d = 0, given unit normal n and points x.
    Returns d, residuals (n^T x - d), slack (max r^2), stats dict.
    """
    n = n_cam / (np.linalg.norm(n_cam) + 1e-12)
    proj = points_cam @ n  # shape (N,)

    if method == "ls":
        d = float(np.mean(proj))
    elif method == "median":
        d = float(np.median(proj))  # robust L1 location in 1D
    elif method == "huber":
        d = float(np.mean(proj))
        for _ in range(20):
            r = proj - d
            w = huber_weights(r, huber_delta)
            d_new = float(np.sum(w * proj) / (np.sum(w) + 1e-12))
            if abs(d_new - d) < 1e-10:
                break
            d = d_new
    else:
        raise ValueError("method must be 'ls', 'median', or 'huber'")

    residuals = proj - d
    stats = {
        "d_meters": d,
        "mean_abs_mm": 1000.0 * float(np.mean(np.abs(residuals))),
        "p98_abs_mm": 1000.0 * float(np.percentile(np.abs(residuals), 98)),
        "max_abs_mm": 1000.0 * float(np.max(np.abs(residuals))),
        "std_mm":     1000.0 * float(np.std(residuals)),
        "num_points": int(points_cam.shape[0]),
    }
    slack = float(np.max(residuals ** 2))  # max r^2 (paper-style)
    return d, residuals, slack, stats

def angle_between_unit_vectors(u, v):
    uu = u / (np.linalg.norm(u) + 1e-12)
    vv = v / (np.linalg.norm(v) + 1e-12)
    c = np.clip(uu @ vv, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

# =======================
# Geometry for TOP plane from SIDE polygon
# =======================
def estimate_side_normal(depth0, side_surface_polygon_xy, intrinsics):
    """PCA normal of the *side* surface (from its image polygon)."""
    H, W = depth0.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [side_surface_polygon_xy.astype(np.int32)], 1)
    rows, cols = np.where(mask == 1)
    pix = np.vstack([cols, rows]).T
    side_cloud = unproject_points(pix, depth0, intrinsics)
    C, x_s, y_s, z_s = pca_frame(side_cloud)
    if C is None:
        raise RuntimeError("Side surface cloud too small to estimate a plane.")
    n_side = z_s / (np.linalg.norm(z_s) + 1e-12)  # unit
    return n_side, side_cloud

def find_top_edge_pixels(side_surface_polygon_xy, y_tol_pixels=3):
    """
    Get the top boundary pixels of the polygon (smallest y in image coords)
    with a tolerance.
    """
    poly = side_surface_polygon_xy.astype(np.int32)
    mask = np.zeros((int(np.max(poly[:,1])+5), int(np.max(poly[:,0])+5)), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    # find external contour
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # OpenCV 3/4 compatibility
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not cnts:
        raise RuntimeError("No contour found for side surface polygon.")
    contour = cnts[0][:, 0, :]  # (M,2) as (x,y)
    y_min = contour[:, 1].min()
    top_edge_2d = contour[np.abs(contour[:, 1] - y_min) <= max(1, y_tol_pixels)]
    if top_edge_2d.shape[0] < 3:
        # widen tolerance if needed
        top_edge_2d = contour[np.abs(contour[:, 1] - y_min) <= 5]
    if top_edge_2d.shape[0] < 3:
        raise RuntimeError("Top edge too sparse; increase y_tol_pixels or check polygon.")
    return top_edge_2d

def estimate_top_normal_from_side(depth0, side_surface_polygon_xy, intrinsics, y_tol_pixels=3):
    """
    1) Get side-plane normal n_side,
    2) Unproject top-edge pixels -> 3D, get edge direction e_edge via PCA,
    3) n_top = edge x n_side (unit).
    """
    n_side, _ = estimate_side_normal(depth0, side_surface_polygon_xy, intrinsics)
    top_edge_2d = find_top_edge_pixels(side_surface_polygon_xy, y_tol_pixels=y_tol_pixels)
    top_edge_3d = unproject_points(top_edge_2d, depth0, intrinsics)
    C, x1, _, _ = pca_frame(top_edge_3d)   # x1 = principal direction along edge
    e_edge = x1 / (np.linalg.norm(x1) + 1e-12)
    n_top = np.cross(e_edge, n_side)
    n_top = n_top / (np.linalg.norm(n_top) + 1e-12)
    return n_side, e_edge, n_top, top_edge_3d

# =======================
# Trowel tip extraction per frame
# =======================
def extract_trowel_tip_points(all_depth_maps, trowel_vertices_2d_traj, intrinsics,
                              tip_mode, triangle_edge_size=0.15):
    """
    Returns:
      tip_points_cam: (M,3)
      frames_used:    (M,)
      tilt_deg:       (M,) angle between trowel surface normal (local 'z' from PCA) and TOP normal (filled later)
      z_normals_cam:  (M,3) raw trowel 'z' axes (for tilt later)
    """
    N, H, W = all_depth_maps.shape
    tip_points, frames_used, z_normals = [], [], []

    for i in range(N):
        verts2d = trowel_vertices_2d_traj[i]
        if verts2d.size == 0:
            continue
        # 3D cloud of the trowel polygon in frame i
        m = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(m, [verts2d.astype(np.int32)], 1)
        rr, cc = np.where(m == 1)
        pix = np.vstack([cc, rr]).T
        cloud = unproject_points(pix, all_depth_maps[i], intrinsics)
        if cloud.shape[0] < 10:
            continue

        # Local PCA frame for trowel in this frame
        C_t, x_t, y_t, z_t = pca_frame(cloud)
        if C_t is None:
            continue

        if tip_mode.lower() == "extreme":
            # extreme along -x_t: choose point with min projection on x_t
            rel = cloud - C_t
            xproj = rel @ (x_t / (np.linalg.norm(x_t) + 1e-12))
            idx = int(np.argmin(xproj))
            p_tip = cloud[idx]
        else:
            # canonical tip at (-L/2, 0, 0) in local
            R_t = np.column_stack([x_t, y_t, z_t])
            tip_local = np.array([-triangle_edge_size / 2.0, 0.0, 0.0])
            p_tip = C_t + R_t @ tip_local

        tip_points.append(p_tip)
        frames_used.append(i)
        z_normals.append(z_t / (np.linalg.norm(z_t) + 1e-12))

    if len(tip_points) == 0:
        raise RuntimeError("No valid frames with trowel polygon available.")
    return (np.asarray(tip_points),
            np.asarray(frames_used, dtype=np.int32),
            np.asarray(z_normals))

# =======================
# Main
# =======================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    all_depth_maps = np.load(DEPTH_MAP_PATH)                           # (N,H,W)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    side_polygon_xy = np.load(SIDE_SURFACE_POLY_PATH, allow_pickle=True)

    if side_polygon_xy.ndim == 3:
        # if saved as shape (1, M, 2)
        side_polygon_xy = side_polygon_xy[0]

    N, H, W = all_depth_maps.shape

    # Build TOP normal from SIDE polygon (on depth[0])
    n_side, e_edge, n_top, top_edge_3d = estimate_top_normal_from_side(
        all_depth_maps[0], side_polygon_xy, INTRINSICS, y_tol_pixels=3
    )

    # Extract tip points (camera frame)
    tip_points_cam, frames_used, trowel_z_normals = extract_trowel_tip_points(
        all_depth_maps, trowel_vertices_2d_traj, INTRINSICS,
        tip_mode=TIP_MODE, triangle_edge_size=TRIANGLE_EDGE_SIZE
    )

    if tip_points_cam.shape[0] < 3:
        raise RuntimeError("Not enough tip points to fit the plane offset.")

    # Fit n_top^T x - d = 0 for tip points
    d_top, residuals, slack, stats = fit_offset_known_normal(
        tip_points_cam, n_top, method=FIT_METHOD, huber_delta=HUBER_DELTA
    )

    # Optional tilt (angle between trowel local z and TOP normal)
    tilt_deg = np.array([angle_between_unit_vectors(z_t, n_top) for z_t in trowel_z_normals],
                        dtype=np.float64)

    # Save outputs
    plane_pack = {"n": n_top.astype(float), "d": float(d_top)}
    # np.save(os.path.join(SAVE_DIR, "plane_top_cam.npy"), plane_pack)
    # np.save(os.path.join(SAVE_DIR, "tip_points_cam.npy"), tip_points_cam)
    # np.save(os.path.join(SAVE_DIR, "residuals.npy"), residuals)
    # np.save(os.path.join(SAVE_DIR, "tilt_degrees.npy"), tilt_deg)
    # np.save(os.path.join(SAVE_DIR, "frames_used.npy"), frames_used)

    # Human-readable JSON stats
    summary = {
        "fit_method": FIT_METHOD,
        "huber_delta_m": float(HUBER_DELTA),
        "tip_mode": TIP_MODE,
        "top_plane": {"n": np2py(n_top), "d": float(d_top)},
        "side_plane": {"n_side": np2py(n_side)},
        "edge_direction": np2py(e_edge),
        "residual_stats": {
            "mean_abs_mm": float(stats["mean_abs_mm"]),
            "p98_abs_mm":  float(stats["p98_abs_mm"]),
            "max_abs_mm":  float(stats["max_abs_mm"]),
            "std_mm":      float(stats["std_mm"]),
            "num_points":  int(stats["num_points"]),
        },
        "tilt_stats_deg": {
            "mean": float(np.mean(tilt_deg)) if tilt_deg.size else None,
            "p98":  float(np.percentile(tilt_deg, 98)) if tilt_deg.size else None,
            "max":  float(np.max(tilt_deg)) if tilt_deg.size else None,
            "std":  float(np.std(tilt_deg)) if tilt_deg.size else None,
        },
        "num_frames_used": int(tip_points_cam.shape[0]),
        "frames_used": np2py(frames_used),
    }
    with open(os.path.join(SAVE_DIR, "stats.txt"), "w") as f:
        f.write(json.dumps(summary, default=np2py, indent=2))

    # Console summary
    print("\n=== TOP-plane point-on-plane fit (camera frame) ===")
    print(f"n_top (unit): {n_top}")
    print(f"d_top (m):    {d_top:.6f}    -> plane:  n_top^T x - d_top = 0")
    print(f"Residuals: mean|r|={stats['mean_abs_mm']:.2f} mm, "
          f"p98|r|={stats['p98_abs_mm']:.2f} mm, max|r|={stats['max_abs_mm']:.2f} mm, "
          f"std={stats['std_mm']:.2f} mm, N={stats['num_points']}")
    print(f"Tilt (deg): mean={np.mean(tilt_deg):.2f}, "
          f"p98={np.percentile(tilt_deg, 98):.2f}, "
          f"max={np.max(tilt_deg):.2f}, std={np.std(tilt_deg):.2f}")
    print(f"Slack (max r^2): {slack:.6e}")
    

if __name__ == "__main__":
    main()
