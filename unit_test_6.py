
"""
Estimate the brick TOP surface plane (normal from SIDE-surface PCA's 2nd axis)
and fit a point-on-plane constraint for the trowel tip trajectory. 
Also visualize to see check.
"""

# import os
# import json
# import numpy as np
# import cv2

# # =======================
# # USER OPTIONS
# # =======================
# TIP_MODE      = "canonical"    # 'extreme' or 'canonical'
# FIT_METHOD    = "huber"        # 'ls' | 'median' | 'huber'
# HUBER_DELTA   = 1e-3           # meters (~1 mm) for Huber
# SAVE_DIR      = "."            # output folder

# # For canonical trowel tip (only if TIP_MODE='canonical')
# TRIANGLE_EDGE_SIZE = 0.15      # meters (tip at (-L/2, 0, 0) in local)

# # Camera intrinsics (fx, fy, cx, cy)
# FX, FY, CX, CY = 836.0, 836.0, 979.0, 632.0
# INTRINSICS = np.array([FX, FY, CX, CY], dtype=np.float64)

# # Input paths
# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
# TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
# SIDE_SURFACE_POLY_PATH  = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# # =======================
# # Helpers
# # =======================
# def np2py(obj):
#     """Make NumPy types JSON-serializable."""
#     import numpy as _np
#     if isinstance(obj, _np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
#         return obj.item()
#     return obj

# def unproject_points(coords_2d, depth_map, intrinsics):
#     """Unproject (x,y) pixels to camera 3D using per-pixel depth (meters)."""
#     fx, fy, cx, cy = intrinsics
#     coords_2d = np.asarray(coords_2d, dtype=np.int32)
#     h, w = depth_map.shape[:2]
#     x_pix = np.clip(coords_2d[:, 0], 0, w - 1)
#     y_pix = np.clip(coords_2d[:, 1], 0, h - 1)
#     Z = depth_map[y_pix, x_pix]
#     ok = np.isfinite(Z) & (Z > 0)
#     x_pix, y_pix, Z = x_pix[ok], y_pix[ok], Z[ok]
#     X = (x_pix - cx) * Z / fx
#     Y = (y_pix - cy) * Z / fy
#     return np.stack([X, Y, Z], axis=-1)

# def pca_frame(points):
#     """
#     PCA frame (camera coords) from a 3D point cloud:
#       returns centroid and (x,y,z) principal axes (orthonormal, right-handed).
#     """
#     P = np.asarray(points)
#     if P.shape[0] < 3:
#         return None, None, None, None
#     C = P.mean(axis=0)
#     Q = P - C
#     cov = np.cov(Q, rowvar=False)
#     w, V = np.linalg.eigh(cov)           # ascending eigenvalues
#     order = np.argsort(w)[::-1]          # descending
#     x = V[:, order[0]]                   # 1st axis (max variance)
#     y = V[:, order[1]]                   # 2nd axis
#     x = x / (np.linalg.norm(x) + 1e-12)
#     y = y - (x @ y) * x
#     y = y / (np.linalg.norm(y) + 1e-12)
#     z = np.cross(x, y)
#     nz = np.linalg.norm(z)
#     if nz < 1e-12:
#         R0 = np.column_stack([x, y, V[:, order[2]]])
#         U, _, Vt = np.linalg.svd(R0, full_matrices=False)
#         R = U @ Vt
#     else:
#         z = z / nz
#         R = np.column_stack([x, y, z])
#     if np.linalg.det(R) < 0:
#         y = -y
#         z = np.cross(x, y); z /= (np.linalg.norm(z) + 1e-12)
#         R = np.column_stack([x, y, z])
#     return C, R[:, 0], R[:, 1], R[:, 2]

# def huber_weights(r, delta):
#     a = np.abs(r)
#     w = np.ones_like(r)
#     mask = a > delta
#     w[mask] = delta / (a[mask] + 1e-12)
#     return w

# def fit_offset_known_normal(points_cam, n_cam, method="ls", huber_delta=1e-3):
#     """Fit d in plane n^T x - d = 0 given unit normal n and points x."""
#     n = n_cam / (np.linalg.norm(n_cam) + 1e-12)
#     proj = points_cam @ n
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
#             if abs(d_new - d) < 1e-10: break
#             d = d_new
#     else:
#         raise ValueError("method must be 'ls', 'median', or 'huber'")
#     residuals = proj - d
#     stats = {
#         "d_meters": d,
#         "mean_abs_mm": 1000.0 * float(np.mean(np.abs(residuals))),
#         "p98_abs_mm":  1000.0 * float(np.percentile(np.abs(residuals), 98)),
#         "max_abs_mm":  1000.0 * float(np.max(np.abs(residuals))),
#         "std_mm":      1000.0 * float(np.std(residuals)),
#         "num_points":  int(points_cam.shape[0]),
#     }
#     slack = float(np.max(residuals ** 2))
#     return d, residuals, slack, stats

# # def angle_between_unit_vectors(u, v):
# #     uu = u / (np.linalg.norm(u) + 1e-12)
# #     vv = v / (np.linalg.norm(v) + 1e-12)
# #     c = np.clip(uu @ vv, -1.0, 1.0)
# #     return float(np.degrees(np.arccos(c)))

# # =======================
# # Side-surface PCA -> use 2nd axis as TOP normal (your “green axis”)
# # =======================
# def top_normal_from_side_pca(depth0, side_surface_polygon_xy, intrinsics):
#     """
#     Compute PCA on the side-surface cloud.
#     Return:
#       n_side  : side-plane normal (smallest-variance axis)
#       n_top   : USE 2ND PCA AXIS as the TOP-surface normal (per your setup)
#       axes    : (x_s, y_s, z_s) = (1st, 2nd, 3rd) PCA axes
#     """
#     H, W = depth0.shape[:2]
#     mask = np.zeros((H, W), dtype=np.uint8)
#     cv2.fillPoly(mask, [side_surface_polygon_xy.astype(np.int32)], 1)
#     rows, cols = np.where(mask == 1)
#     pix = np.vstack([cols, rows]).T
#     cloud = unproject_points(pix, depth0, intrinsics)

#     C, x_s, y_s, z_s = pca_frame(cloud)
#     if C is None:
#         raise RuntimeError("Side surface cloud too small to estimate PCA axes.")

#     # Normalize
#     x_s /= (np.linalg.norm(x_s) + 1e-12)
#     y_s /= (np.linalg.norm(y_s) + 1e-12)
#     z_s /= (np.linalg.norm(z_s) + 1e-12)

#     # In your dataset: y_s (green) is the vertical/edge direction,
#     # which you treat as the TOP-surface normal.
#     n_top  = y_s
#     n_side = z_s
#     return n_side, n_top, (x_s, y_s, z_s), cloud

# # =======================
# # Trowel tip extraction per frame
# # =======================
# def extract_trowel_tip_points(all_depth_maps, trowel_vertices_2d_traj, intrinsics,
#                               tip_mode, triangle_edge_size=0.15):
#     """
#     Returns:
#       tip_points_cam: (M,3)
#       frames_used:    (M,)
#       z_normals_cam:  (M,3) trowel 'z' axes (for tilt later)
#     """
#     N, H, W = all_depth_maps.shape
#     tip_points, frames_used, z_normals = [], [], []

#     for i in range(N):
#         verts2d = trowel_vertices_2d_traj[i]
#         if verts2d.size == 0:
#             continue
#         m = np.zeros((H, W), dtype=np.uint8)
#         cv2.fillPoly(m, [verts2d.astype(np.int32)], 1)
#         rr, cc = np.where(m == 1)
#         pix = np.vstack([cc, rr]).T
#         cloud = unproject_points(pix, all_depth_maps[i], intrinsics)
#         if cloud.shape[0] < 10:
#             continue

#         C_t, x_t, y_t, z_t = pca_frame(cloud)
#         if C_t is None:
#             continue

#         if tip_mode.lower() == "extreme":
#             rel = cloud - C_t
#             xproj = rel @ (x_t / (np.linalg.norm(x_t) + 1e-12))
#             idx = int(np.argmin(xproj))
#             p_tip = cloud[idx]
#         else:
#             R_t = np.column_stack([x_t, y_t, z_t])
#             tip_local = np.array([-triangle_edge_size / 2.0, 0.0, 0.0])
#             p_tip = C_t + R_t @ tip_local

#         tip_points.append(p_tip)
#         frames_used.append(i)
#         z_normals.append(z_t / (np.linalg.norm(z_t) + 1e-12))

#     if len(tip_points) == 0:
#         raise RuntimeError("No valid frames with trowel polygon available.")
#     return (np.asarray(tip_points),
#             np.asarray(frames_used, dtype=np.int32),
#             np.asarray(z_normals))

# # =======================
# # Main
# # =======================
# def main():
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # Load data
#     all_depth_maps = np.load(DEPTH_MAP_PATH)                           # (N,H,W)
#     trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
#     side_polygon_xy = np.load(SIDE_SURFACE_POLY_PATH, allow_pickle=True)
#     if side_polygon_xy.ndim == 3:
#         side_polygon_xy = side_polygon_xy[0]

#     # --- TOP normal from SIDE PCA (use 2nd axis directly, per your convention) ---
#     n_side, n_top, (x_s, y_s, z_s), side_cloud = top_normal_from_side_pca(
#         all_depth_maps[0], side_polygon_xy, INTRINSICS
#     )

#     # --- Extract tip points (camera frame) ---
#     tip_points_cam, frames_used, trowel_z_normals = extract_trowel_tip_points(
#         all_depth_maps, trowel_vertices_2d_traj, INTRINSICS,
#         tip_mode=TIP_MODE, triangle_edge_size=TRIANGLE_EDGE_SIZE
#     )
#     if tip_points_cam.shape[0] < 3:
#         raise RuntimeError("Not enough tip points to fit the plane offset.")

#     # --- Fit n_top^T x - d = 0 ---
#     d_top, residuals, slack, stats = fit_offset_known_normal(
#         tip_points_cam, n_top, method=FIT_METHOD, huber_delta=HUBER_DELTA
#     )

#     # --- Optional tilt (trowel 'z' vs top normal) ---
#     # tilt_deg = np.array([angle_between_unit_vectors(z_t, n_top) for z_t in trowel_z_normals],
#     #                     dtype=np.float64)

#     # --- Save outputs ---
#     summary = {
#         "fit_method": FIT_METHOD,
#         "huber_delta_m": float(HUBER_DELTA),
#         "tip_mode": TIP_MODE,
#         "top_plane": {"n": np2py(n_top), "d": float(d_top)},
#         "side_plane": {"n_side": np2py(n_side)},
#         "side_pca_axes": {"x": np2py(x_s), "y": np2py(y_s), "z": np2py(z_s)},
#         "residual_stats": {
#             "mean_abs_mm": float(stats["mean_abs_mm"]),
#             "p98_abs_mm":  float(stats["p98_abs_mm"]),
#             "max_abs_mm":  float(stats["max_abs_mm"]),
#             "std_mm":      float(stats["std_mm"]),
#             "num_points":  int(stats["num_points"]),
#         },
#         # "tilt_stats_deg": {
#         #     "mean": float(np.mean(tilt_deg)) if tilt_deg.size else None,
#         #     "p98":  float(np.percentile(tilt_deg, 98)) if tilt_deg.size else None,
#         #     "max":  float(np.max(tilt_deg)) if tilt_deg.size else None,
#         #     "std":  float(np.std(tilt_deg)) if tilt_deg.size else None,
#         # },
#         "num_frames_used": int(tip_points_cam.shape[0]),
#         "frames_used": np2py(frames_used),
#     }
#     # with open(os.path.join(SAVE_DIR, "stats.txt"), "w") as f:
#     #     f.write(json.dumps(summary, default=np2py, indent=2))

#     print("\n=== TOP-plane point-on-plane fit (camera frame) ===")
#     print(f"n_top (unit, from SIDE PCA 2nd axis): {n_top}")
#     print(f"d_top (m):    {d_top:.6f}    -> plane:  n_top^T x - d_top = 0")
#     print(f"Residuals: mean|r|={stats['mean_abs_mm']:.2f} mm, "
#           f"p98|r|={stats['p98_abs_mm']:.2f} mm, max|r|={stats['max_abs_mm']:.2f} mm, "
#           f"std={stats['std_mm']:.2f} mm, N={stats['num_points']}")
#     # print(f"Tilt (deg): mean={np.mean(tilt_deg):.2f}, "
#     #       f"p98={np.percentile(tilt_deg, 98):.2f}, "
#     #       f"max={np.max(tilt_deg):.2f}, std={np.std(tilt_deg):.2f}")
#     # print(f"Slack (max r^2): {slack:.6e}")

# if __name__ == "__main__":
#     main()













# import os
# import json
# import numpy as np
# import cv2


# # =======================
# # USER OPTIONS
# # =======================
# TIP_MODE      = "canonical"    # 'extreme' or 'canonical'
# FIT_METHOD    = "huber"        # 'ls' | 'median' | 'huber'
# HUBER_DELTA   = 1e-3           # meters (~1 mm) for Huber
# SAVE_DIR      = "."            # output folder
# VIZ           = True           # <— turn on 3D viz

# # For canonical trowel tip (only if TIP_MODE='canonical')
# TRIANGLE_EDGE_SIZE = 0.15      # meters (tip at (-L/2, 0, 0) in local)

# # Top-plane patch size just for display (length × width on the plane)
# TOP_PLANE_LEN = 0.85           # along x_s (brick length)
# TOP_PLANE_WID = 0.16           # along direction orthogonal to (x_s, n_top) in the plane

# # Camera intrinsics (fx, fy, cx, cy)
# FX, FY, CX, CY = 836.0, 836.0, 979.0, 632.0
# INTRINSICS = np.array([FX, FY, CX, CY], dtype=np.float64)

# # Input paths
# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
# TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
# SIDE_SURFACE_POLY_PATH  = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")

# # =======================
# # Helpers
# # =======================
# def np2py(obj):
#     """Make NumPy types JSON-serializable."""
#     import numpy as _np
#     if isinstance(obj, _np.ndarray):
#         return obj.tolist()
#     if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
#         return obj.item()
#     return obj

# def unproject_points(coords_2d, depth_map, intrinsics):
#     """Unproject (x,y) pixels to camera 3D using per-pixel depth (meters)."""
#     fx, fy, cx, cy = intrinsics
#     coords_2d = np.asarray(coords_2d, dtype=np.int32)
#     h, w = depth_map.shape[:2]
#     x_pix = np.clip(coords_2d[:, 0], 0, w - 1)
#     y_pix = np.clip(coords_2d[:, 1], 0, h - 1)
#     Z = depth_map[y_pix, x_pix]
#     ok = np.isfinite(Z) & (Z > 0)
#     x_pix, y_pix, Z = x_pix[ok], y_pix[ok], Z[ok]
#     X = (x_pix - cx) * Z / fx
#     Y = (y_pix - cy) * Z / fy
#     return np.stack([X, Y, Z], axis=-1)

# def pca_frame(points):
#     """
#     PCA frame (camera coords) from a 3D point cloud:
#       returns centroid and (x,y,z) principal axes (orthonormal, right-handed).
#     """
#     P = np.asarray(points)
#     if P.shape[0] < 3:
#         return None, None, None, None
#     C = P.mean(axis=0)
#     Q = P - C
#     cov = np.cov(Q, rowvar=False)
#     w, V = np.linalg.eigh(cov)           # ascending eigenvalues
#     order = np.argsort(w)[::-1]          # descending
#     x = V[:, order[0]]                   # 1st axis (max variance)
#     y = V[:, order[1]]                   # 2nd axis
#     x = x / (np.linalg.norm(x) + 1e-12)
#     y = y - (x @ y) * x
#     y = y / (np.linalg.norm(y) + 1e-12)
#     z = np.cross(x, y)
#     nz = np.linalg.norm(z)
#     if nz < 1e-12:
#         R0 = np.column_stack([x, y, V[:, order[2]]])
#         U, _, Vt = np.linalg.svd(R0, full_matrices=False)
#         R = U @ Vt
#     else:
#         z = z / nz
#         R = np.column_stack([x, y, z])
#     if np.linalg.det(R) < 0:
#         y = -y
#         z = np.cross(x, y); z /= (np.linalg.norm(z) + 1e-12)
#         R = np.column_stack([x, y, z])
#     return C, R[:, 0], R[:, 1], R[:, 2]

# def huber_weights(r, delta):
#     a = np.abs(r)
#     w = np.ones_like(r)
#     mask = a > delta
#     w[mask] = delta / (a[mask] + 1e-12)
#     return w

# def fit_offset_known_normal(points_cam, n_cam, method="ls", huber_delta=1e-3):
#     """Fit d in plane n^T x - d = 0 given unit normal n and points x."""
#     n = n_cam / (np.linalg.norm(n_cam) + 1e-12)
#     proj = points_cam @ n
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
#             if abs(d_new - d) < 1e-10: break
#             d = d_new
#     else:
#         raise ValueError("method must be 'ls', 'median', or 'huber'")
#     residuals = proj - d
#     stats = {
#         "d_meters": d,
#         "mean_abs_mm": 1000.0 * float(np.mean(np.abs(residuals))),
#         "p98_abs_mm":  1000.0 * float(np.percentile(np.abs(residuals), 98)),
#         "max_abs_mm":  1000.0 * float(np.max(np.abs(residuals))),
#         "std_mm":      1000.0 * float(np.std(residuals)),
#         "num_points":  int(points_cam.shape[0]),
#     }
#     slack = float(np.max(residuals ** 2))
#     return d, residuals, slack, stats



# # =======================
# # Side-surface PCA -> use 2nd axis as TOP normal (your “green axis”)
# # =======================
# def top_normal_from_side_pca(depth0, side_surface_polygon_xy, intrinsics):
#     """
#     Compute PCA on the side-surface cloud.
#     Return:
#       n_side  : side-plane normal (smallest-variance axis)
#       n_top   : USE 2ND PCA AXIS as the TOP-surface normal (per your setup)
#       axes    : (x_s, y_s, z_s) = (1st, 2nd, 3rd) PCA axes
#       C_s     : centroid of the side cloud (used for viz)
#     """
#     H, W = depth0.shape[:2]
#     mask = np.zeros((H, W), dtype=np.uint8)
#     cv2.fillPoly(mask, [side_surface_polygon_xy.astype(np.int32)], 1)
#     rows, cols = np.where(mask == 1)
#     pix = np.vstack([cols, rows]).T
#     cloud = unproject_points(pix, depth0, intrinsics)

#     C, x_s, y_s, z_s = pca_frame(cloud)
#     if C is None:
#         raise RuntimeError("Side surface cloud too small to estimate PCA axes.")

#     x_s /= (np.linalg.norm(x_s) + 1e-12)
#     y_s /= (np.linalg.norm(y_s) + 1e-12)
#     z_s /= (np.linalg.norm(z_s) + 1e-12)

#     n_top  = y_s          # your "green" axis
#     n_side = z_s          # side-plane normal
#     return n_side, n_top, (x_s, y_s, z_s), cloud, C

# # =======================
# # Trowel tip extraction per frame
# # =======================
# def extract_trowel_tip_points(all_depth_maps, trowel_vertices_2d_traj, intrinsics,
#                               tip_mode, triangle_edge_size=0.15):
#     """
#     Returns:
#       tip_points_cam: (M,3)
#       frames_used:    (M,)
#       z_normals_cam:  (M,3) trowel 'z' axes (for tilt later)
#       centroids_cam:  (M,3) simple centroids per frame (for path viz)
#     """
#     N, H, W = all_depth_maps.shape
#     tip_points, frames_used, z_normals, cents = [], [], [], []

#     for i in range(N):
#         verts2d = trowel_vertices_2d_traj[i]
#         if getattr(verts2d, "size", 0) == 0:
#             continue
#         m = np.zeros((H, W), dtype=np.uint8)
#         cv2.fillPoly(m, [verts2d.astype(np.int32)], 1)
#         rr, cc = np.where(m == 1)
#         pix = np.vstack([cc, rr]).T
#         cloud = unproject_points(pix, all_depth_maps[i], intrinsics)
#         if cloud.shape[0] < 10:
#             continue

#         C_t, x_t, y_t, z_t = pca_frame(cloud)
#         if C_t is None:
#             continue

#         if tip_mode.lower() == "extreme":
#             rel = cloud - C_t
#             xproj = rel @ (x_t / (np.linalg.norm(x_t) + 1e-12))
#             idx = int(np.argmin(xproj))
#             p_tip = cloud[idx]
#         else:
#             R_t = np.column_stack([x_t, y_t, z_t])
#             tip_local = np.array([-triangle_edge_size / 2.0, 0.0, 0.0])
#             p_tip = C_t + R_t @ tip_local

#         tip_points.append(p_tip)
#         frames_used.append(i)
#         z_normals.append(z_t / (np.linalg.norm(z_t) + 1e-12))
#         cents.append(C_t)

#     if len(tip_points) == 0:
#         raise RuntimeError("No valid frames with trowel polygon available.")
#     return (np.asarray(tip_points),
#             np.asarray(frames_used, dtype=np.int32),
#             np.asarray(z_normals),
#             np.asarray(cents))

# # =======================
# # Visualization
# # =======================
# def visualize_3d(side_cloud, C_s, x_s, y_s, z_s, n_top, d_top,
#                  tip_points_cam, residuals, centroids_cam):
#     try:
#         import pyvista as pv
#     except Exception as e:
#         print("PyVista not available; skipping viz.", e)
#         return

#     plot = pv.Plotter(window_size=[1200, 720])
#     plot.set_background("white")

#     # Side cloud
#     plot.add_points(side_cloud, color="#888888", point_size=3, render_points_as_spheres=True)

#     # Axes at side centroid
#     arrow_len = 0.12
#     plot.add_arrows(C_s.reshape(1,3), x_s.reshape(1,3), mag=arrow_len, color="red")    # x_s
#     plot.add_arrows(C_s.reshape(1,3), y_s.reshape(1,3), mag=arrow_len, color="green")  # y_s (your n_top)
#     plot.add_arrows(C_s.reshape(1,3), z_s.reshape(1,3), mag=arrow_len, color="blue")   # z_s

#     # Top plane patch (centered near median of tip points, oriented by n_top)
#     center = np.median(tip_points_cam, axis=0)
#     # Use x_s as in-plane axis; compute the other in-plane axis orthogonal to x_s and n_top
#     e1 = x_s / (np.linalg.norm(x_s) + 1e-12)
#     e2 = np.cross(n_top, e1)
#     e2 = e2 / (np.linalg.norm(e2) + 1e-12)

#     L, W = TOP_PLANE_LEN/2.0, TOP_PLANE_WID/2.0
#     pts = np.array([
#         center - L*e1 - W*e2,
#         center + L*e1 - W*e2,
#         center + L*e1 + W*e2,
#         center - L*e1 + W*e2,
#     ])
#     faces = np.hstack([[4, 0,1,2,3]])
#     plane_patch = pv.PolyData(pts, faces)
#     plot.add_mesh(plane_patch, color="#D18B47", opacity=0.55, show_edges=True, edge_color="black")

#     # Tip points
#     plot.add_points(tip_points_cam, color="#1f77b4", point_size=9, render_points_as_spheres=True)

#     # Residual segments (tip -> projection onto plane)
#     n = n_top / (np.linalg.norm(n_top) + 1e-12)
#     lines_pts = []
#     lines_cells = []
#     cell_id = 0
#     for p, r in zip(tip_points_cam, residuals):
#         p_proj = p - r * n
#         lines_pts.append(p); lines_pts.append(p_proj)
#         lines_cells.extend([2, cell_id, cell_id+1])  # '2' means line with 2 points
#         cell_id += 2
#     if lines_pts:
#         lines_pts = np.asarray(lines_pts)
#         lines_cells = np.asarray(lines_cells)
#         line_mesh = pv.PolyData(lines_pts)
#         line_mesh.lines = lines_cells
#         plot.add_mesh(line_mesh, color="magenta", line_width=2)

#     # Trowel centroid path (optional)
#     if centroids_cam.size > 0:
#         path = pv.Spline(centroids_cam, 1000)
#         plot.add_mesh(path, color="navy", line_width=4)

#     plot.add_legend(labels=[
#         ("Side PCA x (length)", "red"),
#         ("Top normal (your PCA y)", "green"),
#         ("Side normal (PCA z)", "blue"),
#         ("Top plane patch", "#D18B47"),
#         ("Tip points", "#1f77b4"),
#         ("Residuals", "magenta"),
#         ("Trowel path", "navy"),
#     ])
#     plot.camera.zoom(1.2)
#     plot.show()

# # =======================
# # Main
# # =======================
# def main():
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     # Load data
#     all_depth_maps = np.load(DEPTH_MAP_PATH)                           # (N,H,W)
#     trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
#     side_polygon_xy = np.load(SIDE_SURFACE_POLY_PATH, allow_pickle=True)
#     if side_polygon_xy.ndim == 3:
#         side_polygon_xy = side_polygon_xy[0]

#     # --- TOP normal from SIDE PCA (use 2nd axis directly, per your convention) ---
#     n_side, n_top, (x_s, y_s, z_s), side_cloud, C_s = top_normal_from_side_pca(
#         all_depth_maps[0], side_polygon_xy, INTRINSICS
#     )

#     # --- Extract tip points (camera frame) ---
#     tip_points_cam, frames_used, trowel_z_normals, centroids_cam = extract_trowel_tip_points(
#         all_depth_maps, trowel_vertices_2d_traj, INTRINSICS,
#         tip_mode=TIP_MODE, triangle_edge_size=TRIANGLE_EDGE_SIZE
#     )
#     if tip_points_cam.shape[0] < 3:
#         raise RuntimeError("Not enough tip points to fit the plane offset.")

#     # --- Fit n_top^T x - d = 0 ---
#     d_top, residuals, slack, stats = fit_offset_known_normal(
#         tip_points_cam, n_top, method=FIT_METHOD, huber_delta=HUBER_DELTA
#     )

    

#     # --- Print summary ---
#     print("\n=== TOP-plane point-on-plane fit (camera frame) ===")
#     print(f"n_top (unit, from SIDE PCA 2nd axis): {n_top}")
#     print(f"d_top (m):    {d_top:.6f}    -> plane:  n_top^T x - d_top = 0")
#     print(f"Residuals: mean|r|={stats['mean_abs_mm']:.2f} mm, "
#           f"p98|r|={stats['p98_abs_mm']:.2f} mm, max|r|={stats['max_abs_mm']:.2f} mm, "
#           f"std={stats['std_mm']:.2f} mm, N={stats['num_points']}")
    
#     # --- Viz ---
#     if VIZ:
#         visualize_3d(side_cloud, C_s, x_s, y_s, z_s, n_top, d_top,
#                      tip_points_cam, residuals, centroids_cam)

# if __name__ == "__main__":
#     main()












#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static 3D viz:
- Brick wall box (from PCA on side surface polygon).
- Canonical trowel triangle at each frame.
- Point-on-plane fitted to the triangle TIP points.
"""

import os
import numpy as np
import pyvista as pv
import cv2
from tqdm import tqdm

# ------------ camera intrinsics ------------
FX = 836.0
FY = 836.0
CX = 979.0
CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY], dtype=np.float64)

# ------------ I/O paths ------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")


# ------------ geometry params ------------
TRIANGLE_EDGE_SIZE = 0.15      # triangle length (m); width is L/2
RECTANGLE_LENGTH   = 0.85      # brick extent along wall x (m)
RECTANGLE_HEIGHT   = 0.04      # brick height along wall y (m)
RECTANGLE_WIDTH    = 0.16      # brick thickness along wall z (m)

# ------------ helpers ------------
def unproject_points(coords_2d, depth_map, intrinsics):
    fx, fy, cx, cy = intrinsics
    coords_2d = np.asarray(coords_2d, dtype=np.int32)
    h, w = depth_map.shape[:2]
    x = np.clip(coords_2d[:, 0], 0, w - 1)
    y = np.clip(coords_2d[:, 1], 0, h - 1)
    Z = depth_map[y, x]
    ok = np.isfinite(Z) & (Z > 0)
    x, y, Z = x[ok], y[ok], Z[ok]
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)

def calculate_local_frame(point_cloud):
    if point_cloud.shape[0] < 3:
        return None, None, None, None
    centroid = np.mean(point_cloud, axis=0)
    centered = point_cloud - centroid
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    v1, v2, v3 = vecs[:, idx[0]], vecs[:, idx[1]], vecs[:, idx[2]]
    # enforce orthonormal RH frame
    x_axis = v1 / (np.linalg.norm(v1) + 1e-12)
    y_axis = v2 - (x_axis @ v2) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    # consistent flip if you prefer
    z_axis = -z_axis
    return centroid, x_axis, y_axis, z_axis

def compute_pose_matrix(centroid, x_axis, y_axis, z_axis):
    R = np.column_stack((x_axis/np.linalg.norm(x_axis),
                         y_axis/np.linalg.norm(y_axis),
                         z_axis/np.linalg.norm(z_axis)))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = centroid
    return T

def fit_offset_known_normal(points_cam, n_cam, method="huber", huber_delta=1e-3):
    """Fit d in plane n^T x - d = 0 given unit normal n and points x."""
    n = n_cam / (np.linalg.norm(n_cam) + 1e-12)
    proj = points_cam @ n
    if method == "ls":
        d = float(np.mean(proj))
    elif method == "median":
        d = float(np.median(proj))
    else:  # huber (1D IRLS)
        d = float(np.mean(proj))
        for _ in range(20):
            r = proj - d
            a = np.abs(r)
            w = np.ones_like(a)
            mask = a > huber_delta
            w[mask] = huber_delta / (a[mask] + 1e-12)
            d_new = float(np.sum(w * proj) / (np.sum(w) + 1e-12))
            if abs(d_new - d) < 1e-10:
                break
            d = d_new
    residuals = proj - d
    return d, residuals

# ------------ main ------------
if __name__ == "__main__":

    # load inputs
    all_depth_maps = np.load(DEPTH_MAP_PATH)                         # (N,H,W)
    trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    brick_wall_vertices_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)

    num_frames, height, width = all_depth_maps.shape

    # ---------- wall side surface -> PCA frame ----------
    brick_wall_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(brick_wall_mask, [brick_wall_vertices_2d.astype(np.int32)], 1)
    rows, cols = np.where(brick_wall_mask == 1)
    wall_pix = np.vstack((cols, rows)).T
    wall_cloud = unproject_points(wall_pix, all_depth_maps[0], INTRINSICS)

    centroid, x_axis, y_axis, z_axis = calculate_local_frame(wall_cloud)
    wall_pose = compute_pose_matrix(centroid, x_axis, y_axis, z_axis)
    
    # convention: y_axis (green) is YOUR top-surface normal; z_axis is side normal
    n_top = y_axis / (np.linalg.norm(y_axis) + 1e-12)

    # ---------- extract trowel local frames + canonical triangles ----------
    trowel_local_frames = []
    trowel_poses_trajectory = []
    tips_cam = []   # canonical tip points in camera frame
    cents_cam = []  # triangle centroids (for path)
    for i in tqdm(range(num_frames), desc="Calculating Trowel Frames"):
        poly = trowel_vertices_2d_traj[i]
        if getattr(poly, "size", 0) == 0:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        rr, cc = np.where(mask == 1)
        pix = np.vstack((cc, rr)).T
        cloud = unproject_points(pix, all_depth_maps[i], INTRINSICS)
        if cloud.shape[0] < 10:
            continue
        C, xt, yt, zt = calculate_local_frame(cloud)
        if C is None:
            continue
        trowel_local_frames.append((C, xt, yt, zt))
        T = compute_pose_matrix(C, xt, yt, zt)
        trowel_poses_trajectory.append(T)
        # canonical triangle tip at (-L/2, 0, 0) in local coords
        tip_local = np.array([-TRIANGLE_EDGE_SIZE/2.0, 0.0, 0.0])
        tip_cam = C + np.column_stack((xt, yt, zt)) @ tip_local
        tips_cam.append(tip_cam)
        cents_cam.append(C)

    tips_cam = np.asarray(tips_cam)
    cents_cam = np.asarray(cents_cam)
    

    # ---------- fit point-on-plane to tips ----------
    d_top, residuals = fit_offset_known_normal(tips_cam, n_top, method="huber", huber_delta=1e-3)

    # ---------- build a display plane parallel to top surface ----------
    # choose a center on the plane near the mean projected tip
    n = n_top / (np.linalg.norm(n_top) + 1e-12)
    proj_vals = tips_cam @ n
    # center of projections (on plane)
    center_on_plane = np.mean(tips_cam - np.outer(proj_vals - d_top, n), axis=0)

    # construct two in-plane axes to make a rectangle patch
    e1 = x_axis / (np.linalg.norm(x_axis) + 1e-12)  # along wall length
    e2 = np.cross(n, e1); e2 = e2 / (np.linalg.norm(e2) + 1e-12)  # completes plane basis

    L = RECTANGLE_LENGTH/2.0
    W = RECTANGLE_WIDTH/2.0
    top_pts = np.array([
        center_on_plane - L*e1 - W*e2,
        center_on_plane + L*e1 - W*e2,
        center_on_plane + L*e1 + W*e2,
        center_on_plane - L*e1 + W*e2,
    ])
    top_faces = np.hstack([[4, 0,1,2,3]])

    # ---------- begin PyVista ----------
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    # wall axes (centroid + arrows)
    arrow_scale = 0.1
    plotter.add_points(centroid, color='black', point_size=10, label='Wall Centroid')
    plotter.add_arrows(cent=np.array([centroid]), direction=np.array([x_axis]), mag=arrow_scale, color='red',   label='Wall X-Axis')
    plotter.add_arrows(cent=np.array([centroid]), direction=np.array([y_axis]), mag=arrow_scale, color='green', label='Wall Y-Axis (Top normal)')
    plotter.add_arrows(cent=np.array([centroid]), direction=np.array([z_axis]), mag=arrow_scale, color='blue',  label='Wall Z-Axis (Side normal)')

    # wall box (volume)
    l, w, h = RECTANGLE_LENGTH/2, RECTANGLE_WIDTH/2, RECTANGLE_HEIGHT/2
    v1_2d, v2_2d, v3_2d, v4_2d = np.array([-l, -2*h]), np.array([l, -2*h]), np.array([l, 0]), np.array([-l, 0])

    v1f = centroid + v1_2d[0]*x_axis + v1_2d[1]*y_axis
    v2f = centroid + v2_2d[0]*x_axis + v2_2d[1]*y_axis
    v3f = centroid + v3_2d[0]*x_axis + v3_2d[1]*y_axis
    v4f = centroid + v4_2d[0]*x_axis + v4_2d[1]*y_axis

    v1b = v1f + w*z_axis
    v2b = v2f + w*z_axis
    v3b = v3f + w*z_axis
    v4b = v4f + w*z_axis

    box_pts = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
    box_faces = np.hstack([
        [4, 0,1,2,3],
        [4, 4,5,6,7],
        [4, 0,1,5,4],
        [4, 1,2,6,5],
        [4, 2,3,7,6],
        [4, 3,0,4,7],
    ]).astype(np.int64)
    box_mesh = pv.PolyData(box_pts, box_faces)
    plotter.add_mesh(box_mesh, color='#D95319', opacity=1, show_edges=True, edge_color='black', line_width=3, label='Brick Wall')

    # top constraint plane (semi-transparent)
    plane_mesh = pv.PolyData(top_pts, top_faces)
    plotter.add_mesh(plane_mesh, color='#8FD19E', opacity=0.55, show_edges=True, edge_color='black', label='Point-on-plane')

    # trowel triangles along the trajectory
    if len(trowel_local_frames):
        cent_path = np.array([f[0] for f in trowel_local_frames])
        plotter.add_mesh(pv.Spline(cent_path, 1000), color="blue", line_width=5, label="Trowel Centroid Path")
        plotter.add_points(cent_path[0], color='green', point_size=15, render_points_as_spheres=True, label='Start')
        plotter.add_points(cent_path[-1], color='red', point_size=15, render_points_as_spheres=True, label='End')

        for i, (C, xt, yt, zt) in enumerate(trowel_local_frames):
            Ltri = TRIANGLE_EDGE_SIZE
            Wtri = Ltri / 2.0
            v1_2 = np.array([-Ltri/2, 0])         # TIP in local 2D (x,y)
            v2_2 = np.array([ Ltri/2, -Wtri/2])
            v3_2 = np.array([ Ltri/2,  Wtri/2])
            v1_3 = C + v1_2[0]*xt + v1_2[1]*yt
            v2_3 = C + v2_2[0]*xt + v2_2[1]*yt
            v3_3 = C + v3_2[0]*xt + v3_2[1]*yt
            tri_pts = np.array([v1_3, v2_3, v3_3])
            tri_face = np.hstack([3, 0,1,2])
            tri_mesh = pv.PolyData(tri_pts, tri_face)
            plotter.add_mesh(tri_mesh, color='grey', opacity=0.8, show_edges=True)

    # show tip points and residual segments
    plotter.add_points(tips_cam, color='navy', point_size=10, render_points_as_spheres=True, label='Tip points')
    n = n_top / (np.linalg.norm(n_top) + 1e-12)
    # build polylines tip -> projection
    if tips_cam.size:
        pts_lines = []
        cells = []
        cid = 0
        for p, r in zip(tips_cam, residuals):
            p_proj = p - r * n
            pts_lines.extend([p, p_proj])
            cells.extend([2, cid, cid+1])  # line with 2 points
            cid += 2
        lines = pv.PolyData(np.asarray(pts_lines))
        lines.lines = np.asarray(cells)
        plotter.add_mesh(lines, color='magenta', line_width=2, label='Residuals')

    plotter.add_legend()
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    plotter.show()

    # quick text summary
    mean_abs_mm = 1e3 * float(np.mean(np.abs(residuals)))
    print(f"\nPlane fit with n_top (green): d = {float(np.mean(tips_cam @ n) - d_top):.6f} (sanity)")
    print(f"Residuals: mean|r| = {mean_abs_mm:.2f} mm, N = {len(residuals)}")
