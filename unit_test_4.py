#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end:
1) Load depth maps, polygons, plane mask, and saved tool poses (4x4).
2) Unproject and visualize wall cloud + best-fit plane.
3) Build a demo 3D centroid trajectory from trowel polygons (camera frame).
4) Learn a single-demo ProMP and condition it (start/mid/end or all points).
5) Generate a new trajectory.
6) Visualize the generated path and place a canonical triangle at subsampled
   points, using orientations interpolated (SLERP) from the saved poses.

Dependencies: numpy, opencv-python, pyvista, matplotlib, tqdm (optional)
"""

import os
import numpy as np
import cv2
import pyvista as pv
from tqdm import tqdm

# ------------------------- Config (edit as needed) -------------------------

# Camera intrinsics
FX = 836.0
FY = 836.0
CX = 979.0
CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY], dtype=np.float64)

# Paths (assume this script sits in project root; change if needed)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPTH_MAP_PATH            = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH   = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")            # (T,) object array, each (Ni,2)
TROWEL_TIP_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_tip_polygon_vertices.npy")      # optional
BRICK_WALL_VERTICES_PATH  = os.path.join(PROJECT_ROOT, "brick_wall_side_surface.npy")            # (M,2) int32 polygon
POSES_PATH                = os.path.join(PROJECT_ROOT, "trowel_poses_trajectory.npy")            # (T,4,4)
# If you already have a prebuilt (T,4) [phase, x, y, z] demo, you can point to it. Otherwise we compute from polygons.
PREBUILT_DEMO_PATH        = None

# Visualization parameters
TRIANGLE_EDGE_SIZE = 0.15  # meters (long side of canonical triangle)
RECTANGLE_LENGTH   = 0.85  # meters
RECTANGLE_HEIGHT = 0.04      # in meters for the brick wall plane
RECTANGLE_WIDTH = 0.16
WALL_POINT_SIZE    = 3

# ProMP hyperparameters
K_BASIS    = 30
SIGMA_RBF  = 1.0 / K_BASIS        # width in canonical time
RIDGE_LS   = 1e-6
LAMBDA2    = 1e-7                 # prior weight covariance (lambda^2 * I)
SIGMA_C2   = 1e-6                 # conditioning noise (smaller = harder)
USE_ALL_DEMO_POINTS = False       # True to "memorize" via conditioning on all points
N_SAMPLES  = 0                    # extra sampled trajectories (0 for none)

# Triangle glyph density along the generated curve
N_TRIANGLES_APPROX = 50



def to_cv_polys(poly_like, width, height):
    """Coerce a single polygon or list-of-polygons to a list of (M,2) int32 arrays within image bounds."""
    def _coerce_one(arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        arr = np.round(arr).astype(np.int32)
        arr[:, 0] = np.clip(arr[:, 0], 0, width - 1)
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
    """2D pixels -> 3D points (camera frame) using per-pixel depth."""
    fx, fy, cx, cy = intrinsics
    coords_2d = np.asarray(coords_2d)
    if coords_2d.ndim == 1:
        coords_2d = coords_2d.reshape(-1, 2)
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
    """Return (centroid, x_axis, y_axis, z_axis) from PCA; z is smallest-variance direction (flipped to point toward camera if desired)."""
    if point_cloud.shape[0] < 3:
        return None, None, None, None
    centroid = point_cloud.mean(axis=0)
    Xc = point_cloud - centroid
    cov = np.cov(Xc, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    x_axis = evecs[:, idx[0]]
    y_axis = evecs[:, idx[1]]
    z_axis = evecs[:, idx[2]]  # normal
    z_axis = -z_axis           # optional flip (keep from earlier code)
    return centroid, x_axis, y_axis, z_axis



def centroid_from_polygon_mask(poly_2d, depth, intrinsics):
    """Rasterize polygon to a mask, unproject its pixels, return 3D centroid."""
    h, w = depth.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_2d], 1)
    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return None, None  # centroid, cloud
    pixels = np.stack((xs, ys), axis=1)
    cloud = unproject_points(pixels, depth, intrinsics)
    if cloud.shape[0] == 0:
        return None, None
    return cloud.mean(axis=0), cloud


def demo_from_polygons(depth_maps, body_polys, intrinsics):
    """
    Build a (T,4) demo array: [phase, x, y, z] using 3D centroids of body polygons.
    Skips frames with no valid cloud; phase is linear over retained frames.
    """
    T_all, H, W = depth_maps.shape
    traj_pts = []
    kept_idx = []
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
    traj_pts = np.vstack(traj_pts)  # (T,3)
    # canonical time over the kept frames
    T = traj_pts.shape[0]
    s = np.linspace(0.0, 1.0, T)
    demo = np.column_stack([s, traj_pts])
    return demo, np.array(kept_idx, dtype=np.int32)


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

# ------------------------- Pose interpolation & triangle glyphs -------------------------

def canonical_triangle(length=0.15, width=None):
    if width is None:
        width = length / 2.0
    v1 = np.array([-length/2,  0.0,       0.0])
    v2 = np.array([ length/2, -width/2.,  0.0])
    v3 = np.array([ length/2,  width/2.,  0.0])
    V  = np.stack([v1, v2, v3], axis=0)
    V -= V.mean(axis=0, keepdims=True)
    return V

def quaternion_from_matrix(R):
    m = R
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    else:
        i = np.argmax([m[0,0], m[1,1], m[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / s
            x = 0.25 * s; y = (m[0,1] + m[1,0]) / s; z = (m[0,2] + m[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s; y = 0.25 * s; z = (m[1,2] + m[2,1]) / s
        else:
            s = np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s; y = (m[1,2] + m[2,1]) / s; z = 0.25 * s
    q = np.array([w,x,y,z], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return q


def slerp(q0, q1, u):
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1; dot = -dot
    DOT_THRESH = 0.9995
    if dot > DOT_THRESH:
        q = q0 + u*(q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * u
    s0 = np.sin(theta_0 - theta) / (sin_0 + 1e-12)
    s1 = np.sin(theta) / (sin_0 + 1e-12)
    return s0*q0 + s1*q1


def R_from_quat(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)]
    ], dtype=np.float64)


def interp_pose_rotations(Ts, u):
    """SLERP rotations in Ts (N,4,4) at phase u in [0,1]."""
    N = Ts.shape[0]
    if N == 1:
        return Ts[0,:3,:3]
    s = u * (N - 1)
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, N - 1)
    a  = s - i0
    R0 = Ts[i0,:3,:3]; R1 = Ts[i1,:3,:3]
    q0 = quaternion_from_matrix(R0); q1 = quaternion_from_matrix(R1)
    q  = slerp(q0, q1, a)
    return R_from_quat(q)


def draw_wall_cloud_and_plane(plotter, wall_poly_2d, depth_map, intrinsics):
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
    # plotter.add_points(cloud, style='points', color='#D95319', render_points_as_spheres=True,
    #                    point_size=WALL_POINT_SIZE, label='Brick Wall Cloud')
    centroid, x_axis, y_axis, z_axis = pca_frame(cloud)
    if centroid is None:
        return None
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

    l, h, w = RECTANGLE_LENGTH / 2, RECTANGLE_HEIGHT / 2, RECTANGLE_WIDTH / 2
    v1_2d, v2_2d, v3_2d, v4_2d = np.array([-l, -2*h]), np.array([l, -2*h]), np.array([l, 0]), np.array([-l, 0])
    # v1_3d = C + v1_2d[0] * x_axis + v1_2d[1] * y_axis
    # v2_3d = C + v2_2d[0] * x_axis + v2_2d[1] * y_axis
    # v3_3d = C + v3_2d[0] * x_axis + v3_2d[1] * y_axis
    # v4_3d = C + v4_2d[0] * x_axis + v4_2d[1] * y_axis
    # rectangle_points = np.array([v1_3d, v2_3d, v3_3d, v4_3d])
    # face = np.hstack([4, 0, 1, 2, 3])
    # rectangle_mesh = pv.PolyData(rectangle_points, face)
    # plotter.add_mesh(rectangle_mesh, 
    #                  color='#D95319', 
    #                  opacity=0.8, 
    #                  show_edges=True, 
    #                  edge_color='black', 
    #                  line_width=3, 
    #                  label='Wall Best-Fit Plane')
    

    # Front (+w) and back (-w) faces
    v1f = centroid + v1_2d[0]*x_axis + v1_2d[1]*y_axis
    v2f = centroid + v2_2d[0]*x_axis + v2_2d[1]*y_axis
    v3f = centroid + v3_2d[0]*x_axis + v3_2d[1]*y_axis
    v4f = centroid + v4_2d[0]*x_axis + v4_2d[1]*y_axis

    v1b = centroid + v1_2d[0]*x_axis + v1_2d[1]*y_axis + w*z_axis
    v2b = centroid + v2_2d[0]*x_axis + v2_2d[1]*y_axis + w*z_axis
    v3b = centroid + v3_2d[0]*x_axis + v3_2d[1]*y_axis + w*z_axis
    v4b = centroid + v4_2d[0]*x_axis + v4_2d[1]*y_axis + w*z_axis

    # Vertex order: 0..3 = front (ccw when looking from +z_axis), 4..7 = back
    pts = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])

    # 6 quad faces (each starts with the number of indices = 4)
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
        label='Wall Volume (thickened plane)'
    )



    return cloud



def main():
    
    depth_maps = np.load(DEPTH_MAP_PATH)  # (T, H, W)
    H, W = depth_maps.shape[1:]
    body_polys = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)  # (T,) object array
    poses = np.load(POSES_PATH)  # (T_pose, 4, 4)
    wall_poly_2d = np.load(BRICK_WALL_VERTICES_PATH, allow_pickle=True)  # (M,2)

    print(f"depth_maps: {depth_maps.shape}, body_polys: {body_polys.shape}, poses: {poses.shape}")

    # --- Build demo (phase, x, y, z) ---
    if PREBUILT_DEMO_PATH and os.path.exists(PREBUILT_DEMO_PATH):
        demo = np.load(PREBUILT_DEMO_PATH)  # (T,4)
        xs = demo[:,0]; Yd = demo[:,1:4]
        print(f"Loaded prebuilt demo: {demo.shape}")
    else:
        demo, kept = demo_from_polygons(depth_maps, body_polys, INTRINSICS)  # (T,4)
        xs = demo[:,0]; Yd = demo[:,1:4]
        print(f"Built demo from polygons: {demo.shape} (kept {len(kept)} frames)")

    T = len(xs); D = 3

    # --- Learn ProMP ---
    mu_w, alpha2, Phi = fit_weights_single_demo(demo, K_BASIS, SIGMA_RBF, RIDGE_LS)

    # --- Constraints (either all demo points or a subset) ---
    if USE_ALL_DEMO_POINTS:
        xs_c = xs.copy()
        ys_c = Yd.copy()
    else:
        idxs = [0, max(1,T//4), max(2,T//2), max(3,3*T//4), T-1]
        xs_c = xs[idxs]
        ys_c = Yd[idxs, :]

    mu_w_cond, Sigma_w_cond = condition_on_points(
        mu_w, LAMBDA2, xs_c, ys_c, K_BASIS, D, SIGMA_RBF, SIGMA_C2
    )

    # --- Generate using mean (and optional samples) ---
    gen_curve = predict_mean_at(xs, K_BASIS, mu_w_cond, D, SIGMA_RBF)  # (T,3)
    curves = [gen_curve]
    if N_SAMPLES > 0:
        rng = np.random.default_rng(0)
        for _ in range(N_SAMPLES):
            curves.append(sample_trajectory(xs, K_BASIS, mu_w_cond, Sigma_w_cond, alpha2, D, SIGMA_RBF, rng))

    # --- Visualization: wall + demo + generated path + triangles with demo orientations ---
    plotter = pv.Plotter(window_size=[1200, 800])
    plotter.set_background('white')

    print("Projecting and plotting wall cloud...")
    _ = draw_wall_cloud_and_plane(plotter, wall_poly_2d, depth_maps[0], INTRINSICS)

    # Plot demo curve
    plotter.add_mesh(pv.Spline(Yd, 1000), color="pink", line_width=2, label="Demo (ref)")
    plotter.add_points(Yd[0], color='gray', point_size=12, render_points_as_spheres=True)
    plotter.add_points(Yd[-1], color='gray', point_size=12, render_points_as_spheres=True)

    # Plot generated mean path
    plotter.add_mesh(pv.Spline(gen_curve, 1000), color="royalblue", line_width=4, label="Generated Path")
    plotter.add_points(gen_curve[0], color='green', point_size=15, render_points_as_spheres=True, label='Start')
    plotter.add_points(gen_curve[-1], color='red', point_size=15, render_points_as_spheres=True, label='End')

    # Optional sample paths
    for i, C in enumerate(curves[1:], start=1):
        plotter.add_mesh(pv.Spline(C, 1000), color="orange", line_width=1.5, label=f"Sample {i}")

    # Place canonical triangle glyphs along generated mean using pose orientations
    V_local = canonical_triangle(length=TRIANGLE_EDGE_SIZE, width=TRIANGLE_EDGE_SIZE/2)
    T_gen   = gen_curve.shape[0]
    T_pose  = poses.shape[0]
    step    = max(1, T_gen // max(1, N_TRIANGLES_APPROX))

    for i in range(0, T_gen, step):
        p = gen_curve[i]  # generated centroid position (camera frame)
        u = i / max(1, T_gen - 1)
        R = interp_pose_rotations(poses, u)     # orientation from demo at the same phase
        V_world = (R @ V_local.T).T + p[None, :]
        face = np.hstack([3, 0, 1, 2])
        tri = pv.PolyData(V_world, face)
        opacity = 0.25 + 0.65 * (i / (T_gen - 1))
        plotter.add_mesh(tri, 
                         color='grey', 
                         opacity=0.8, 
                         show_edges=True)

    plotter.add_legend()
    plotter.camera.azimuth = -60
    plotter.camera.elevation = 25
    plotter.camera.zoom(1.3)

    print("\nShowing plot... rotate/zoom with mouse.")
    plotter.show()



if __name__ == "__main__":
    main()
