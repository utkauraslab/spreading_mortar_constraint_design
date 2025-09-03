
# """
# Visualize trowel's relative pose in the brick-wall local frame.

# Inputs (same as before):
# - depth_map_cross_frames_refined.npy    : (N, H, W) depth maps
# - trowel_polygon_vertices.npy           : lper-frame polygon vertices in image pixels (x,y)
# - brick_wall_side_surface.npy           : single polygon vertices of the wall side surface (x,y) in a reference frame (use frame 0 depth)

# Coordinate transforms:
# - Camera -> Wall: p_w = R_wc^T @ (p_c - t_wc)
# """

# import os
# import sys
# import numpy as np
# import pyvista as pv
# import cv2
# from tqdm import tqdm


# FX = 836.0
# FY = 836.0
# CX = 979.0
# CY = 632.0
# INTRINSICS = np.array([FX, FY, CX, CY])


# PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
# TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
# BRICK_WALL_VERTICES_PATH = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")



# TRIANGLE_EDGE_SIZE  = 0.15
# RECTANGLE_LENGTH    = 0.85   # along wall x
# RECTANGLE_HEIGHT    = 0.04   # along wall y (we'll draw 2*height tall band)
# RECTANGLE_WIDTH     = 0.16   # along wall z (thickness)


# def unproject_points(coords_2d, depth_map, intrinsics):
#     """
#     2D pixels -> 3D camera coordinates. coords_2d should be (M,2) in (x,y) with integer pixel indices.
#     """
#     fx, fy, cx, cy = intrinsics
#     if not isinstance(coords_2d, np.ndarray):
#         coords_2d = np.array(coords_2d)
#     x_coords, y_coords = coords_2d[:, 0].astype(int), coords_2d[:, 1].astype(int)

#     # Guard against out-of-bounds due to polygon edges touching borders
#     H, W = depth_map.shape[:2]
#     mask_in = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
#     x_coords = x_coords[mask_in]
#     y_coords = y_coords[mask_in]
#     if x_coords.size == 0:
#         return np.empty((0, 3), dtype=float)

#     Z = depth_map[y_coords, x_coords]
#     valid_mask = (Z > 0) & ~np.isnan(Z)
#     x_coords, y_coords, Z = x_coords[valid_mask], y_coords[valid_mask], Z[valid_mask]
#     if Z.size == 0:
#         return np.empty((0, 3), dtype=float)

#     X = (x_coords - cx) * Z / fx
#     Y = (y_coords - cy) * Z / fy
#     return np.stack((X, Y, Z), axis=-1)


# def calculate_local_frame(point_cloud):
#     """
#     PCA-based local frame from a point cloud.
#     Returns: centroid, x_axis, y_axis, z_axis (all np.ndarray(3,))
#     Guarantees a right-handed frame and consistent normal flipping (z may be flipped to your preference).
#     """
#     if point_cloud.shape[0] < 3:
#         return None, None, None, None

#     centroid = np.mean(point_cloud, axis=0)
#     centered = point_cloud - centroid
#     cov = np.cov(centered, rowvar=False)
#     vals, vecs = np.linalg.eigh(cov)  # eigenvectors are columns
#     idx = np.argsort(vals)[::-1]
#     v1, v2, v3 = vecs[:, idx[0]], vecs[:, idx[1]], vecs[:, idx[2]]

#     # Enforce right-handed frame: z = x × y
#     z_axis = np.cross(v1, v2)
#     if np.linalg.norm(z_axis) < 1e-9:
#         return None, None, None, None
#     z_axis = z_axis / np.linalg.norm(z_axis)

#     x_axis = v1 / np.linalg.norm(v1)
#     # Recompute y to be orthonormal
#     y_axis = np.cross(z_axis, x_axis)
#     y_axis = y_axis / np.linalg.norm(y_axis)

#     z_axis = -z_axis

#     return centroid, x_axis, y_axis, z_axis


# # def calculate_local_frame(point_cloud):
# #     """
# #     PCA method to calculate the local coordinate frame for a point cloud.
# #     """
# #     if point_cloud.shape[0] < 3:
# #         return None, None, None, None
# #     centroid = np.mean(point_cloud, axis=0)
# #     centered_points = point_cloud - centroid
# #     covariance_matrix = np.cov(centered_points, rowvar=False)
# #     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
# #     sorted_indices = np.argsort(eigenvalues)[::-1]
# #     x_axis = eigenvectors[:, sorted_indices[0]]
# #     y_axis = eigenvectors[:, sorted_indices[1]]
# #     z_axis = eigenvectors[:, sorted_indices[2]]
# #     z_axis = -z_axis
# #     return centroid, x_axis, y_axis, z_axis





# def compute_pose_matrix(centroid, x_axis, y_axis, z_axis):
#     """Build 4x4 pose from axes and centroid."""
#     R = np.column_stack((x_axis/np.linalg.norm(x_axis),
#                          y_axis/np.linalg.norm(y_axis),
#                          z_axis/np.linalg.norm(z_axis)))
#     T = np.eye(4, dtype=float)
#     T[:3, :3] = R
#     T[:3, 3]  = centroid
#     return T




# if __name__ == "__main__":
#     # Load data
#     all_depth_maps = np.load(DEPTH_MAP_PATH)
#     trowel_vertices_2d_traj = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
#     brick_wall_vertices_2d  = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)
#     num_frames, height, width = all_depth_maps.shape

    
#     trowel_local_frames = []  # list of tuples: (centroid, x, y, z) in CAMERA coordinates
#     for i in tqdm(range(num_frames), desc="Calculating Trowel Frames"):
#         frame_vertices = trowel_vertices_2d_traj[i]
#         if frame_vertices.size == 0:
#             continue

#         # rasterize polygon to mask
#         mask = np.zeros((height, width), dtype=np.uint8)
#         cv2.fillPoly(mask, [frame_vertices.astype(np.int32)], 1)
#         rows, cols = np.where(mask == 1)
#         if rows.size == 0:
#             continue

#         pixel_coords_2d = np.stack([cols, rows], axis=1)
#         pts3d_cam = unproject_points(pixel_coords_2d, all_depth_maps[i], INTRINSICS)
#         if pts3d_cam.shape[0] < 3:
#             continue

#         local_frame = calculate_local_frame(pts3d_cam)
#         if local_frame[0] is not None:
#             trowel_local_frames.append(local_frame)

    
    
#     wall_mask = np.zeros((height, width), dtype=np.uint8)
#     cv2.fillPoly(wall_mask, [brick_wall_vertices_2d.astype(np.int32)], 1)
#     wr, wc = np.where(wall_mask == 1)
#     wall_pixels = np.stack([wc, wr], axis=1)
#     wall_pts3d_cam = unproject_points(wall_pixels, all_depth_maps[0], INTRINSICS)
#     if wall_pts3d_cam.shape[0] < 3:
#         raise RuntimeError("Wall point cloud too small. Check wall polygon or depth map.")

#     centroid_w_c, x_w_c, y_w_c, z_w_c = calculate_local_frame(wall_pts3d_cam)
#     if centroid_w_c is None:
#         raise RuntimeError("Failed to compute wall local frame.")

#     # Wall pose in CAMERA: ^cT_w
#     T_wall2cam = compute_pose_matrix(centroid_w_c, x_w_c, y_w_c, z_w_c)
#     R_wc = T_wall2cam[:3, :3]  # ^cR_w
#     t_wc = T_wall2cam[:3, 3]   # ^ct_w


#     # camera->wall transform for points
#     def cam_to_wall_points(Pc):
#         """Pc: (N,3) camera points -> (N,3) wall-frame points"""
#         if Pc.ndim == 1:
#             Pc = Pc.reshape(1, 3)
#         return (R_wc.T @ (Pc.T - t_wc.reshape(3, 1))).T


#     def project_to_so3(Rm: np.ndarray) -> np.ndarray:
#         """最邻近 SO(3) 投影，保证正交 + det=+1"""
#         U, _, Vt = np.linalg.svd(Rm)
#         Rproj = U @ Vt
#         if np.linalg.det(Rproj) < 0:
#             U[:, -1] *= -1
#             Rproj = U @ Vt
#         return Rproj

#     # 线性部分：先相机->wall，再按你可视化时的 x 轴取反
#     R_lin = np.eye(3)
#     R_lin = R_wc.T @ R_lin          # camera -> wall
#     S_flip = np.diag([-1, 1, 1])    # 你的 target 帧里做的 x 取反（与位置一致）
#     A = S_flip @ R_lin              # 用这套线性映射来变换“方向向量”

#     poses_wall = []
#     for (cent_c, x_c, y_c, z_c) in trowel_local_frames:
#         # 位置：相机->wall，然后 x 取反（与你现在画路径一致）
#         p_w = cam_to_wall_points(cent_c.reshape(1, 3)).reshape(3)
#         p_w[0] *= -1.0

#         # 姿态：三个局部轴向量用 A 做线性变换
#         xw = A @ (x_c / np.linalg.norm(x_c))
#         yw = A @ (y_c / np.linalg.norm(y_c))

#         # 右手正交化（避免镜像导致的 det<0）
#         xw = xw / np.linalg.norm(xw)
#         yw = yw - xw * np.dot(xw, yw)
#         yw = yw / np.linalg.norm(yw)
#         zw = np.cross(xw, yw)
#         zw = zw / np.linalg.norm(zw)
#         # 重新修正 y，确保严格正交
#         yw = np.cross(zw, xw)

#         R_w = np.column_stack([xw, yw, zw])
#         R_w = project_to_so3(R_w)  # 数值保险

#         # 组装 4x4
#         T = np.eye(4, dtype=float)
#         T[:3, :3] = R_w
#         T[:3, 3]  = p_w
#         poses_wall.append(T)

#     T_wall_seq = np.stack(poses_wall, axis=0)  # (N,4,4)
#     np.save("trowel_pose_in_target_frame.npy", T_wall_seq)
#     print("Saved pose sequence (target frame):", "trowel_pose_in_target_frame.npy", T_wall_seq.shape)



   
   
#     trowel_centroids_cam = np.array([fr[0] for fr in trowel_local_frames])  # (N,3)
#     trowel_centroids_wall = cam_to_wall_points(trowel_centroids_cam)        # (N,3)
#     trowel_centroids_wall[:, 0] = -trowel_centroids_wall[:, 0]  # Flip x to match wall frame convention

    
#     plotter = pv.Plotter(window_size=[1400, 900])
#     plotter.set_background("white")

#     # 1) Draw wall axes at origin of wall frame
#     plotter.add_points(np.zeros((1, 3)), color="black", point_size=12, label="Wall Origin")
#     arrow_scale = 0.1
#     plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[-1, 0, 0]]),
#                        mag=arrow_scale, color="red", label="Wall X-Axis")
#     plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[0, 1, 0]]),
#                        mag=arrow_scale, color="green", label="Wall Y-Axis")
#     plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[0, 0, 1]]),
#                        mag=arrow_scale, color="blue", label="Wall Z-Axis (Normal)")

    
    
#     l, w, h = RECTANGLE_LENGTH / 2, RECTANGLE_WIDTH / 2, RECTANGLE_HEIGHT / 2
#     # front face (z=0 plane band from y=-2h to y=0)
#     v1f = np.array([-l, -2*h, 0])
#     v2f = np.array([+l, -2*h, 0])
#     v3f = np.array([+l,     0, 0])
#     v4f = np.array([-l,     0, 0])
#     # back face (z = +w)
#     v1b, v2b, v3b, v4b = v1f.copy(), v2f.copy(), v3f.copy(), v4f.copy()
#     v1b[2] += w; v2b[2] += w; v3b[2] += w; v4b[2] += w

#     pts_w = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
#     faces = np.hstack([
#         [4, 0, 1, 2, 3],   # front
#         [4, 4, 5, 6, 7],   # back
#         [4, 0, 1, 5, 4],
#         [4, 1, 2, 6, 5],
#         [4, 2, 3, 7, 6],
#         [4, 3, 0, 4, 7],
#     ]).astype(np.int64)
#     wall_mesh = pv.PolyData(pts_w, faces)
#     plotter.add_mesh(
#         wall_mesh, color="#D95319", opacity=1.0,
#         show_edges=True, edge_color="black", line_width=2,
#         label="Wall Volume (wall frame)"
#     )

    
    
#     plotter.add_mesh(pv.Spline(trowel_centroids_wall, 1000),
#                      color="blue", line_width=5, label="Trowel Centroid Path")
#     plotter.add_points(trowel_centroids_wall[0],  color="green", point_size=15,
#                        render_points_as_spheres=True, label="Start")
#     plotter.add_points(trowel_centroids_wall[-1], color="red",   point_size=15,
#                        render_points_as_spheres=True, label="End")

    
    
#     DRAW_TROWEL_TRIANGLES = True
#     if DRAW_TROWEL_TRIANGLES:
#         for i, (cent_c, x_c, y_c, z_c) in enumerate(trowel_local_frames):
#             # Tri in the trowel local x-y plane, centered at centroid
#             length, width_tri = TRIANGLE_EDGE_SIZE, TRIANGLE_EDGE_SIZE / 2
#             v1_2d = np.array([-length/2, 0])
#             v2_2d = np.array([+length/2, -width_tri / 2])
#             v3_2d = np.array([+length/2, +width_tri / 2])

#             # 3D in CAMERA first
#             v1_c = cent_c + v1_2d[0] * x_c + v1_2d[1] * y_c
#             v2_c = cent_c + v2_2d[0] * x_c + v2_2d[1] * y_c
#             v3_c = cent_c + v3_2d[0] * x_c + v3_2d[1] * y_c

#             # Then transform to WALL
#             tri_w = cam_to_wall_points(np.stack([v1_c, v2_c, v3_c], axis=0))
#             tri_w[:, 0] *= -1
#             face = np.hstack([3, 0, 1, 2])
#             tri_mesh = pv.PolyData(tri_w, face)
#             plotter.add_mesh(tri_mesh, color="grey", opacity=0.4, show_edges=True)

#     plotter.add_legend()
#     plotter.camera.azimuth = -60
#     plotter.camera.elevation = 25
#     plotter.camera.zoom(1.3)
#     plotter.show()









"""
Validate & visualize the trowel pose trajectory in the brick-wall (target) local frame.

- Camera frame:  x right, y down, z into the scene.
- Target (wall) frame: axes directions aligned with camera (x right, y down, z into).
- Tool (trowel) local axes from PCA each frame: x = largest eigenvector, y = 2nd, z = normal (smallest).
- ALWAYS return right-handed frames (det=+1). If we flip one axis, we rebuild the others to keep det=+1.

Transform used:
p_w = R_wc^T @ (p_c - t_wc)
R_w = R_wc^T @ R_c
"""

import os
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


TRIANGLE_EDGE_SIZE = 0.15
TRIANGLE_WIDTH = TRIANGLE_EDGE_SIZE / 2
RECTANGLE_LENGTH = 0.85
RECTANGLE_HEIGHT = 0.04
RECTANGLE_WIDTH = 0.16

TRIAD_STRIDE = 8
TRIAD_SCALE  = 0.02   


def add_triad(plotter, T, scale=TRIAD_SCALE, label_prefix=None):
    
    R = T[:3, :3]; p = T[:3, 3]
    x = R[:, 0]; y = R[:, 1]; z = R[:, 2]
    cent = np.array([p])  # (1,3)
    plotter.add_arrows(cent, np.array([x]), mag=scale, color="red",
                       label=f"{label_prefix or ''}x")
    plotter.add_arrows(cent, np.array([y]), mag=scale, color="green",
                       label=f"{label_prefix or ''}y")
    plotter.add_arrows(cent, np.array([z]), mag=scale, color="blue",
                       label=f"{label_prefix or ''}z")
    

def unproject_points(coords_2d, depth_map, intrinsics):
    fx, fy, cx, cy = intrinsics
    coords_2d = np.asarray(coords_2d)
    x_coords, y_coords = coords_2d[:, 0].astype(int), coords_2d[:, 1].astype(int)
    H, W = depth_map.shape[:2]
    mask_in = (x_coords >= 0) & (x_coords < W) & (y_coords >= 0) & (y_coords < H)
    x_coords, y_coords = x_coords[mask_in], y_coords[mask_in]
    if x_coords.size == 0:
        return np.empty((0, 3), float)
    Z = depth_map[y_coords, x_coords]
    valid = (Z > 0) & ~np.isnan(Z)
    x_coords, y_coords, Z = x_coords[valid], y_coords[valid], Z[valid]
    if Z.size == 0:
        return np.empty((0, 3), float)
    X = (x_coords - cx) * Z / fx
    Y = (y_coords - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)

def orthonormalize_R(R):
    """Nearest SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_ = U @ Vt
    if np.linalg.det(R_) < 0:
        U[:, -1] *= -1
        R_ = U @ Vt
    return R_

def make_right_handed(x, y):
    """Given two non-collinear vectors, build a right-handed orthonormal basis (x,y,z)."""
    x = x / (np.linalg.norm(x) + 1e-12)
    y = y - x * np.dot(x, y)
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)
    # ensure det=+1 already; if not, flip z then rebuild y
    if np.linalg.det(np.column_stack([x, y, z])) < 0:
        z = -z
        y = np.cross(z, x)
        y = y / (np.linalg.norm(y) + 1e-12)
    return x, y, z


def pca_local_frame(pts, prefer_z_dir=None):
    """Return centroid and a right-handed (x,y,z) from PCA."""
    if pts.shape[0] < 3:
        return None, None, None, None
    c = np.mean(pts, axis=0)
    Q = pts - c
    C = np.cov(Q, rowvar=False)
    vals, vecs = np.linalg.eigh(C)              # columns are eigenvectors
    idx = np.argsort(vals)[::-1]
    x = vecs[:, idx[0]]
    y = vecs[:, idx[1]]
    x, y, z = make_right_handed(x, y)           # build z = x×y

    # Optional: orient z toward prefer_z_dir
    if prefer_z_dir is not None:
        u = prefer_z_dir / (np.linalg.norm(prefer_z_dir) + 1e-12)
        if np.dot(z, u) < 0:
            z = -z
            y = np.cross(z, x)
            y = y / (np.linalg.norm(y) + 1e-12)
    return c, x, y, z

def compute_pose(centroid, x, y, z):
    R = np.column_stack([x, y, z])
    R = orthonormalize_R(R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centroid
    return T

def cam_to_wall_points(Pc, R_wc, t_wc):
    Pc = np.atleast_2d(Pc)
    Pw = (R_wc.T @ (Pc.T - t_wc.reshape(3, 1))).T
    return Pw


if __name__ == "__main__":
    depth = np.load(DEPTH_MAP_PATH)
    tr_polys = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    wall_poly = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)

    N, H, W = depth.shape

    # -------- tool frames in CAMERA --------
    trowel_frames_cam = []  # list of (centroid, x, y, z)
    for i in tqdm(range(N), desc="Tool PCA per-frame"):
        verts = tr_polys[i]
        if verts.size == 0:
            continue
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [verts.astype(np.int32)], 1)
        rows, cols = np.where(mask == 1)
        if rows.size == 0:
            continue
        pts2d = np.stack([cols, rows], axis=1)
        pts3d = unproject_points(pts2d, depth[i], INTRINSICS)
        if pts3d.shape[0] < 3:
            continue

        # Tool: x=largest, y=second, z=normal (right-handed)
        c, x, y, z = pca_local_frame(pts3d)  # no preferred z for tool
        if c is None:
            continue
        trowel_frames_cam.append((c, x, y, z))

    if len(trowel_frames_cam) == 0:
        raise RuntimeError("No valid tool frames recovered.")

    # -------- wall frame in CAMERA --------
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [wall_poly.astype(np.int32)], 1)
    wr, wc = np.where(mask == 1)
    wall_pts2d = np.stack([wc, wr], axis=1)
    wall_pts3d_cam = unproject_points(wall_pts2d, depth[0], INTRINSICS)
    if wall_pts3d_cam.shape[0] < 3:
        raise RuntimeError("Wall point cloud too small.")

    # Target axes aligned with camera axes (x right, y down, z into):
    # Use PCA for tangent directions, but make z point to camera +z.
    cam_z = np.array([0.0, 0.0, 1.0])
    c_w_c, x_w_c, y_w_c, z_w_c = pca_local_frame(wall_pts3d_cam, prefer_z_dir=cam_z)

    # 让 wall 的 x、y 尽量与 camera 的 +x/+y 同向（仅符号对齐，保持右手性）
    if np.dot(x_w_c, np.array([1.0, 0.0, 0.0])) < 0:
        x_w_c = -x_w_c
    # 重新构造保证右手系
    x_w_c, y_w_c, z_w_c = make_right_handed(x_w_c, y_w_c)
    # 再确保 z 与 camera +z 同向
    if np.dot(z_w_c, cam_z) < 0:
        z_w_c = -z_w_c
        y_w_c = np.cross(z_w_c, x_w_c)
        y_w_c = y_w_c / (np.linalg.norm(y_w_c) + 1e-12)

    T_wall2cam = compute_pose(c_w_c, x_w_c, y_w_c, z_w_c)
    R_wc = T_wall2cam[:3, :3]
    t_wc = T_wall2cam[:3, 3]

    # -------- transform tool poses: CAMERA -> WALL --------
    poses_wall = []
    for (c_c, x_c, y_c, z_c) in trowel_frames_cam:
        # position
        p_w = cam_to_wall_points(c_c, R_wc, t_wc)[0]

        # orientation
        R_c = np.column_stack([
            x_c / (np.linalg.norm(x_c) + 1e-12),
            y_c / (np.linalg.norm(y_c) + 1e-12),
            z_c / (np.linalg.norm(z_c) + 1e-12),
        ])
        R_w = orthonormalize_R(R_wc.T @ R_c)

        T = np.eye(4)
        T[:3, :3] = R_w
        T[:3, 3] = p_w
        poses_wall.append(T)

    T_wall_seq = np.stack(poses_wall, axis=0)
    np.save("trowel_pose_in_target_frame.npy", T_wall_seq)
    print("Saved:", "trowel_pose_in_target_frame.npy", T_wall_seq.shape)

    
    plotter = pv.Plotter(window_size=[1400, 900])
    plotter.set_background("white")

    # axes
    plotter.add_points(np.zeros((1,3)), color="black", point_size=12, label="Wall Origin")
    arrow_scale = 0.12
    plotter.add_arrows(np.array([[0,0,0]]), np.array([[1,0,0]]), mag=arrow_scale, color="red",   label="+X (right)")
    plotter.add_arrows(np.array([[0,0,0]]), np.array([[0,1,0]]), mag=arrow_scale, color="green", label="+Y (down)")
    plotter.add_arrows(np.array([[0,0,0]]), np.array([[0,0,1]]), mag=arrow_scale, color="blue",  label="+Z (into)")

    # wall cuboid (front face z=0, band in y∈[-2h,0])
    l, w, h = RECTANGLE_LENGTH/2, RECTANGLE_WIDTH/2, RECTANGLE_HEIGHT/2
    v1f = np.array([-l, -2*h, 0]); v2f = np.array([+l, -2*h, 0])
    v3f = np.array([+l,     0, 0]); v4f = np.array([-l,     0, 0])
    v1b, v2b, v3b, v4b = v1f.copy(), v2f.copy(), v3f.copy(), v4f.copy()
    v1b[2] += w; v2b[2] += w; v3b[2] += w; v4b[2] += w
    pts = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
    faces = np.hstack([
        [4, 0,1,2,3],
        [4, 4,5,6,7],
        [4, 0,1,5,4],
        [4, 1,2,6,5],
        [4, 2,3,7,6],
        [4, 3,0,4,7],
    ]).astype(np.int64)
    plotter.add_mesh(pv.PolyData(pts, faces),
                     color="#D95319", opacity=1.0,
                     show_edges=True, edge_color="black", line_width=2,
                     label="Brick Wall")

    # path
    P = T_wall_seq[:, :3, 3]
    plotter.add_mesh(pv.Spline(P, 1000), color="#d62728", line_width=5, label="Path")
    plotter.add_points(P[0],  color="green", point_size=16, render_points_as_spheres=True, label="Start")
    plotter.add_points(P[-1], color="red",   point_size=16, render_points_as_spheres=True, label="End")

    for i in range(0, len(T_wall_seq), max(1, TRIAD_STRIDE)):
        add_triad(plotter, T_wall_seq[i], scale=TRIAD_SCALE, label_prefix="tool ")

    # triangles (canonical in local x-y of each pose)
    def add_triangle(T, length=TRIANGLE_EDGE_SIZE, width=TRIANGLE_WIDTH, color="gray", opacity=0.45):
        R = T[:3, :3]; p = T[:3, 3]
        x = R[:, 0]; y = R[:, 1]
        v1 = p + (-length/2)*x + 0.0*y
        v2 = p + ( length/2)*x + (-width/2)*y
        v3 = p + ( length/2)*x + ( width/2)*y
        tri = np.stack([v1, v2, v3], axis=0)
        face = np.hstack([3, 0, 1, 2]).astype(np.int64)
        plotter.add_mesh(pv.PolyData(tri, face), color=color, opacity=opacity, show_edges=True, edge_color="black")

    for i in range(len(T_wall_seq)):
        add_triangle(T_wall_seq[i])

    plotter.add_legend()
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    plotter.show()
