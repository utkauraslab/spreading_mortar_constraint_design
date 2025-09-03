"""
ProMP 生成位置 + 参考 SE(3) 轨迹提供姿态 (SLERP 重采样) → 混合 SE(3) 轨迹与可视化
"""

import os
import numpy as np
import cv2
import pyvista as pv

# ---------------------------- 路径与相机内参 ----------------------------
FX = 836.0; FY = 836.0; CX = 979.0; CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY], dtype=np.float64)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPTH_MAP_PATH              = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")  # (T,H,W)
TROWEL_VERTICES_2D_PATH     = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")          # (T,) object-array，每帧(Ni,2)
BRICK_WALL_VERTICES_PATH    = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")          # (M,2)
POSES_DEMO_PATH             = os.path.join(PROJECT_ROOT, "trowel_poses_trajectory.npy")          # (T_pose,4,4)，仅用于对比/可视化
# 参考 SE(3)（“姿态正确但帧更多”）——从这里拿 orientation 并重采样
REF_POSE_PATH               = os.path.join(PROJECT_ROOT, "generated_pose_se3_K4.npy")            # (N_ref,4,4)

# 如果已有预构建的 (T,4)[phase,x,y,z] 示教 demo，则可直接用；否则从 polygon+depth 生成
PREBUILT_DEMO_PATH          = None

# ---------------------------- 可视化参数 ----------------------------
TRIANGLE_EDGE_SIZE = 0.15
RECTANGLE_LENGTH   = 0.85
RECTANGLE_HEIGHT   = 0.04
RECTANGLE_WIDTH    = 0.16

# ---------------------------- ProMP 超参 ----------------------------
K_BASIS    = 30
SIGMA_RBF  = 1.0 / K_BASIS
RIDGE_LS   = 1e-6
LAMBDA2    = 1e-7
SIGMA_C2   = 1e-6
USE_ALL_DEMO_POINTS = False
N_SAMPLES  = 0           # 额外采样轨迹条数（0 表示只画均值）

# ---------------------------- 工具函数 ----------------------------
def to_cv_polys(poly_like, width, height):
    """将(单个/多个)polygon 转为限定在尺寸内的 int32 轮廓数组列表。"""
    def _coerce_one(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        arr = np.round(arr).astype(np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, width  - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, height - 1)
        return arr

    if isinstance(poly_like, (list, tuple)):
        return [_coerce_one(p) for p in poly_like if np.asarray(p).size >= 6]
    arr = np.asarray(poly_like, dtype=object)
    if arr.dtype == object:
        polys = []
        for p in arr:
            p_arr = np.asarray(p)
            if p_arr.size >= 6:
                polys.append(_coerce_one(p_arr))
        return polys
    else:
        return [_coerce_one(arr)] if arr.size >= 6 else []

def unproject_points(coords_2d, depth_map, intrinsics):
    """按像素深度将 2D 坐标反投影为相机坐标系下 3D 点。"""
    fx, fy, cx, cy = intrinsics
    coords_2d = np.asarray(coords_2d).reshape(-1, 2)
    h, w = depth_map.shape
    xs = coords_2d[:, 0].astype(np.int32)
    ys = coords_2d[:, 1].astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys = xs[valid], ys[valid]
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    Z = depth_map[ys, xs]
    valid_z = (Z > 0) & ~np.isnan(Z)
    xs, ys, Z = xs[valid_z], ys[valid_z], Z[valid_z]
    if Z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1).astype(np.float32)

def pca_frame(point_cloud):
    """返回 (centroid, x_axis, y_axis, z_axis)；z 为最小方差方向（这里取反，面向相机）。"""
    if point_cloud.shape[0] < 3:
        return None, None, None, None
    C = point_cloud.mean(axis=0)
    Xc = point_cloud - C
    cov = np.cov(Xc, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    x_axis = evecs[:, idx[0]]
    y_axis = evecs[:, idx[1]]
    z_axis = -evecs[:, idx[2]]  # 反向
    return C, x_axis, y_axis, z_axis

def centroid_from_polygon_mask(poly_2d, depth, intrinsics):
    """polygon → mask → unproject → 返回 3D centroid 与点云。"""
    h, w = depth.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_2d], 1)
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return None, None
    pixels = np.stack((xs, ys), axis=1)
    cloud = unproject_points(pixels, depth, intrinsics)
    if cloud.shape[0] == 0:
        return None, None
    return cloud.mean(axis=0), cloud

def demo_from_polygons(depth_maps, body_polys, intrinsics):
    """由多帧多边形+深度图生成为 (T,4): [phase, x, y, z] 的单演示 demo。"""
    T_all, H, W = depth_maps.shape
    traj_pts, kept_idx = [], []
    for i in range(T_all):
        arr = np.asarray(body_polys[i])
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
            continue
        c, cloud = centroid_from_polygon_mask(arr.astype(np.int32), depth_maps[i], intrinsics)
        if c is None:
            continue
        traj_pts.append(c)
        kept_idx.append(i)
    if len(traj_pts) == 0:
        raise RuntimeError("No valid 3D centroids from polygons.")
    traj_pts = np.vstack(traj_pts)                # (T,3)
    T = traj_pts.shape[0]
    s = np.linspace(0.0, 1.0, T)
    demo = np.column_stack([s, traj_pts])        # (T,4)
    return demo, np.array(kept_idx, dtype=np.int32)

# ---------------------------- ProMP ----------------------------
def rbf_kernel(K, xt, sigma=None):
    if sigma is None:
        sigma = 1.0 / K
    centers = np.linspace(0.0, 1.0, K)
    v = np.exp(-0.5 * ((xt - centers) ** 2) / (sigma ** 2))
    s = v.sum()
    if s > 1e-12:
        v = v / s
    return v

def build_design_matrix(xs, K, sigma=None):
    Phi = np.zeros((len(xs), K), dtype=np.float64)
    for i, xt in enumerate(xs):
        Phi[i, :] = rbf_kernel(K, xt, sigma)
    return Phi

def build_block_phi(xt, K, D=3, sigma=None):
    phi = rbf_kernel(K, xt, sigma).reshape(1, K)
    return np.kron(np.eye(D), phi)  # (D, K*D)

def build_Phi_constraints(xs_constraints, K, D=3, sigma=None):
    blocks = [build_block_phi(xc, K, D, sigma) for xc in xs_constraints]
    return np.vstack(blocks)  # (M*D, K*D)

def fit_weights_single_demo(demo_traj, K, sigma=None, ridge=1e-6):
    T = demo_traj.shape[0]
    xs = demo_traj[:, 0]
    Y  = demo_traj[:, 1:4]  # (T,3)
    D  = Y.shape[1]

    Phi = build_design_matrix(xs, K, sigma)      # (T,K)
    A   = Phi.T @ Phi + ridge * np.eye(K)
    mu_w = np.zeros((K*D, 1), dtype=np.float64)

    for d in range(D):
        y_d = Y[:, d].reshape(T, 1)
        w_d = np.linalg.solve(A, Phi.T @ y_d)    # (K,1)
        mu_w[d*K:(d+1)*K, :] = w_d

    residuals = np.zeros((T, D))
    for d in range(D):
        w_d = mu_w[d*K:(d+1)*K, :]
        residuals[:, d] = (Y[:, d].reshape(T,1) - Phi @ w_d).ravel()
    dof = max(1, D*T - D*K)
    alpha2 = float(np.sum(residuals**2) / dof)
    return mu_w, alpha2, Phi

def condition_on_points(mu_w, lambda2, xs_constraints, y_constraints, K, D=3, sigma=None, sigma_c2=1e-4):
    xs_constraints = np.asarray(xs_constraints).ravel()
    M = xs_constraints.shape[0]
    assert y_constraints.shape == (M, D)
    Phi_c = build_Phi_constraints(xs_constraints, K, D, sigma)   # (M*D, K*D)
    Y_c   = y_constraints.reshape(M*D, 1)
    Sigma_w = lambda2 * np.eye(K*D)
    R       = sigma_c2 * np.eye(M*D)
    S = Phi_c @ Sigma_w @ Phi_c.T + R
    K_gain = Sigma_w @ Phi_c.T @ np.linalg.solve(S, np.eye(M*D))
    mu_w_cond    = mu_w + K_gain @ (Y_c - Phi_c @ mu_w)
    Sigma_w_cond = Sigma_w - K_gain @ Phi_c @ Sigma_w
    return mu_w_cond, Sigma_w_cond

def predict_mean_at(xs, K, mu_w, D=3, sigma=None):
    xs = np.asarray(xs)
    Tn = xs.shape[0]
    Y  = np.zeros((Tn, D), dtype=np.float64)
    for i, xt in enumerate(xs):
        Phi_D = build_block_phi(xt, K, D, sigma)  # (D, K*D)
        Y[i, :] = (Phi_D @ mu_w).ravel()
    return Y

def sample_trajectory(xs, K, mu_w, Sigma_w, alpha2=0.0, D=3, sigma=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    w_s = rng.multivariate_normal(mu_w.ravel(), Sigma_w).reshape(-1,1)
    Y   = predict_mean_at(xs, K, w_s, D, sigma)
    if alpha2 > 0.0:
        noise = rng.multivariate_normal(np.zeros(D), alpha2*np.eye(D), size=len(xs))
        Y = Y + noise
    return Y

# ---------------------------- 四元数/SLERP 与姿态插值 ----------------------------
def quaternion_from_matrix(Rm):
    m = Rm; t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0;  w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s; y = (m[0,2] - m[2,0]) / s; z = (m[1,0] - m[0,1]) / s
    else:
        i = np.argmax([m[0,0], m[1,1], m[2,2]])
        if   i == 0:
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / s; x = 0.25 * s; y = (m[0,1] + m[1,0]) / s; z = (m[0,2] + m[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / s; x = (m[0,1] + m[1,0]) / s; y = 0.25 * s; z = (m[1,2] + m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / s; x = (m[0,2] + m[2,0]) / s; y = (m[1,2] + m[2,1]) / s; z = 0.25 * s
    q = np.array([w,x,y,z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q

def slerp(q0, q1, u):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1; dot = -dot
    DOT_THRESH = 0.9995
    if dot > DOT_THRESH:
        q = q0 + u * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    th0 = np.arccos(np.clip(dot, -1.0, 1.0)); sin0 = np.sin(th0)
    th  = th0 * u
    s0 = np.sin(th0 - th) / (sin0 + 1e-12)
    s1 = np.sin(th)       / (sin0 + 1e-12)
    return s0*q0 + s1*q1

def R_from_quat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=np.float64)

def interp_pose_rotations(Ts, u):
    """在参考 SE(3) 轨迹 Ts (N,4,4) 上按相位 u∈[0,1] 做 SLERP，返回 3x3 旋转。"""
    N = Ts.shape[0]
    if N == 1:
        return Ts[0,:3,:3]
    s  = u * (N - 1)
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, N - 1)
    a  = s - i0
    R0 = Ts[i0,:3,:3]; R1 = Ts[i1,:3,:3]
    q0 = quaternion_from_matrix(R0); q1 = quaternion_from_matrix(R1)
    q  = slerp(q0, q1, a)
    return R_from_quat(q)

# ---------------------------- 可视化 ----------------------------
def canonical_triangle(length=0.15, width=None):
    if width is None:
        width = length / 2.0
    v1 = np.array([-length/2,  0.0,       0.0])
    v2 = np.array([ length/2, -width/2.,  0.0])
    v3 = np.array([ length/2,  width/2.,  0.0])
    V  = np.stack([v1, v2, v3], axis=0)
    V -= V.mean(axis=0, keepdims=True)
    return V

def draw_wall_cloud_and_plane(plotter, wall_poly_2d, depth_map, intrinsics):
    """画墙体盒（基于 PCA 的中心/法向），可按需替换为散点可视化。"""
    h, w = depth_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [wall_poly_2d.astype(np.int32)], 1)
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return None
    pixels = np.stack((xs, ys), axis=1)
    cloud  = unproject_points(pixels, depth_map, intrinsics)
    if cloud.shape[0] == 0:
        return None
    C, x_axis, y_axis, z_axis = pca_frame(cloud)
    if C is None:
        return None

    l = RECTANGLE_LENGTH/2; h2 = RECTANGLE_HEIGHT/2; w2 = RECTANGLE_WIDTH/2
    v1_2d, v2_2d, v3_2d, v4_2d = np.array([-l, -2*h2]), np.array([l, -2*h2]), np.array([l, 0]), np.array([-l, 0])

    v1f = C + v1_2d[0]*x_axis + v1_2d[1]*y_axis
    v2f = C + v2_2d[0]*x_axis + v2_2d[1]*y_axis
    v3f = C + v3_2d[0]*x_axis + v3_2d[1]*y_axis
    v4f = C + v4_2d[0]*x_axis + v4_2d[1]*y_axis

    v1b = v1f + w2*z_axis; v2b = v2f + w2*z_axis
    v3b = v3f + w2*z_axis; v4b = v4f + w2*z_axis

    pts = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
    faces = np.hstack([
        [4, 0,1,2,3], [4, 4,5,6,7],
        [4, 0,1,5,4], [4, 1,2,6,5],
        [4, 2,3,7,6], [4, 3,0,4,7],
    ]).astype(np.int64)

    box_mesh = pv.PolyData(pts, faces)
    plotter.add_mesh(box_mesh, color='#D95319', opacity=1, show_edges=True,
                     edge_color='black', line_width=2, label='Wall Volume')
    return cloud

# ---------------------------- 主流程 ----------------------------
def main():
    # 载入数据
    depth_maps = np.load(DEPTH_MAP_PATH)                   # (T,H,W)
    body_polys = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)  # (T,) object
    wall_poly_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)  # (M,2)
    poses_demo = np.load(POSES_DEMO_PATH)                  # (T_pose,4,4) 可选，仅用于对比
    poses_ref  = np.load(REF_POSE_PATH)                    # (N_ref,4,4) 用于姿态参考（帧更多）

    print(f"depth_maps: {depth_maps.shape}, body_polys: {body_polys.shape}, poses_demo: {poses_demo.shape}, poses_ref: {poses_ref.shape}")

    # --- 构建/读取 demo: (T,4): [phase, x, y, z] ---
    if PREBUILT_DEMO_PATH and os.path.exists(PREBUILT_DEMO_PATH):
        demo = np.load(PREBUILT_DEMO_PATH)
        xs = demo[:,0]; Yd = demo[:,1:4]
        print(f"Loaded prebuilt demo: {demo.shape}")
    else:
        demo, kept = demo_from_polygons(depth_maps, body_polys, INTRINSICS)
        xs = demo[:,0]; Yd = demo[:,1:4]
        print(f"Built demo from polygons: {demo.shape} (kept {len(kept)} frames)")

    T = len(xs); D = 3

    # --- 学 ProMP ---
    mu_w, alpha2, Phi = fit_weights_single_demo(demo, K_BASIS, SIGMA_RBF, RIDGE_LS)

    # --- 条件化（可选：全部点/少量锚点） ---
    if USE_ALL_DEMO_POINTS:
        xs_c = xs.copy()
        ys_c = Yd.copy()
    else:
        idxs = [0, max(1, T//4), max(2, T//2), max(3, 3*T//4), T-1]
        xs_c = xs[idxs]
        ys_c = Yd[idxs, :]
    mu_w_cond, Sigma_w_cond = condition_on_points(
        mu_w, LAMBDA2, xs_c, ys_c, K_BASIS, D, SIGMA_RBF, SIGMA_C2
    )

    # --- 用均值生成位置（必要时也可采样） ---
    gen_curve = predict_mean_at(xs, K_BASIS, mu_w_cond, D, SIGMA_RBF)   # (T,3)
    curves = [gen_curve]
    if N_SAMPLES > 0:
        rng = np.random.default_rng(0)
        for _ in range(N_SAMPLES):
            curves.append(sample_trajectory(xs, K_BASIS, mu_w_cond, Sigma_w_cond, alpha2, D, SIGMA_RBF, rng))

    # --- 关键：从参考 SE(3) poses_ref 中按相位重采样姿态，得到与 gen_curve 同长度的 R 序列 ---
    # 相位使用 xs 归一化（更通用；若 xs 已是 [0,1]，等同于线性）
    xs_norm = (xs - xs[0]) / (xs[-1] - xs[0] + 1e-12)
    R_list = [interp_pose_rotations(poses_ref, float(u)) for u in xs_norm]  # T × (3,3)

    # --- 组装混合 SE(3)：位置=ProMP，姿态=参考姿态重采样 ---
    T_mix = np.zeros((T, 4, 4), dtype=np.float64)
    T_mix[:, 3, 3] = 1.0
    for i in range(T):
        T_mix[i, :3, :3] = R_list[i]
        T_mix[i, :3,  3] = gen_curve[i]
    out_path = os.path.join(PROJECT_ROOT, "trajectory_promp_pos_refori.npy")
    # np.save(out_path, T_mix)
    print("Saved mixed SE(3) trajectory:", T_mix.shape, "->", out_path)

    # ------------------- 可视化 -------------------
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')
    _ = draw_wall_cloud_and_plane(plotter, wall_poly_2d, depth_maps[0], INTRINSICS)

    # 示教曲线
    plotter.add_mesh(pv.Spline(Yd, 1000), color="pink", line_width=2, label="Demo (ref)")
    plotter.add_points(Yd[0],  color='gray', point_size=12, render_points_as_spheres=True)
    plotter.add_points(Yd[-1], color='gray', point_size=12, render_points_as_spheres=True)

    # 生成位置曲线
    plotter.add_mesh(pv.Spline(gen_curve, 1000), color="royalblue", line_width=4, label="ProMP Position")
    plotter.add_points(gen_curve[0], color='green', point_size=15, render_points_as_spheres=True, label='Start')
    plotter.add_points(gen_curve[-1], color='red',   point_size=15, render_points_as_spheres=True, label='End')

    # 用混合 SE(3) 的姿态画三角形 glyph
    V_local = canonical_triangle(length=TRIANGLE_EDGE_SIZE, width=TRIANGLE_EDGE_SIZE/2)
    N_TRIANGLES_APPROX = 50
    step = max(1, T // max(1, N_TRIANGLES_APPROX))
    face = np.hstack([3, 0, 1, 2])
    for i in range(0, T, step):
        R_i = T_mix[i, :3, :3]
        p_i = T_mix[i, :3,  3]
        V_world = (R_i @ V_local.T).T + p_i[None, :]
        tri = pv.PolyData(V_world, face)
        plotter.add_mesh(tri, color='grey', opacity=0.8, show_edges=True)

    plotter.add_legend()
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)
    plotter.show()

if __name__ == "__main__":
    main()
