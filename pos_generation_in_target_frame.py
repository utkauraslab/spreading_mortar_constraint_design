import numpy as np
import pyvista as pv

# ========= 输入 =========
POSE_PATH = "trowel_pose_in_target_frame.npy"   # (T,4,4)
T_seq = np.load(POSE_PATH)
assert T_seq.ndim == 3 and T_seq.shape[1:] == (4,4), "expect (N,4,4) pose sequence in target frame"

P_demo = T_seq[:, :3, 3]   # (T,3) positions
T_demo = P_demo.shape[0]

# ========= ProMP 基础工具 =========
def rbf_kernel(K, x, sigma=None):
    if sigma is None:
        sigma = 1.0 / K
    centers = np.linspace(0, 1, K)
    return np.exp(-0.5 * ((x - centers) ** 2) / (sigma**2))

def build_design_matrix(xs, K, sigma=None):
    Phi = np.zeros((len(xs), K))
    for i, x in enumerate(xs):
        Phi[i, :] = rbf_kernel(K, x, sigma)
        Phi[i, :] /= Phi[i, :].sum() + 1e-12
    return Phi

def fit_promp_single(P_demo, K=20, sigma=None, ridge=1e-6):
    """Learn mean weight vector from one demo trajectory"""
    T, D = P_demo.shape
    xs = np.linspace(0, 1, T)
    Phi = build_design_matrix(xs, K, sigma)  # (T,K)
    A = Phi.T @ Phi + ridge * np.eye(K)
    mu_w = np.zeros((K*D,1))
    for d in range(D):
        y = P_demo[:,d].reshape(T,1)
        w_d = np.linalg.solve(A, Phi.T @ y)
        mu_w[d*K:(d+1)*K, :] = w_d
    return mu_w, Phi

def predict_mean_at(xs, K, mu_w, D=3, sigma=None):
    """Reconstruct trajectory from learned weights"""
    Y = np.zeros((len(xs), D))
    for i, x in enumerate(xs):
        phi = rbf_kernel(K, x, sigma)
        phi /= phi.sum() + 1e-12
        for d in range(D):
            w_d = mu_w[d*K:(d+1)*K, :]
            Y[i,d] = (phi @ w_d).item()
    return Y

# ========= ProMP 拟合 & 生成 =========
K_BASIS = 50
mu_w, Phi = fit_promp_single(P_demo, K=K_BASIS)

# Conditioning: 保证起点和 demo 一样
xs = np.linspace(0, 1, T_demo)
Y_mean = predict_mean_at(xs, K_BASIS, mu_w)

# 强制替换起点
Y_mean[0,:] = P_demo[0,:]

P_gen = Y_mean   # (T,3)

# ========= 可视化 =========
RECT_LENGTH = 0.85
RECT_HEIGHT = 0.04
RECT_WIDTH  = 0.16

def build_wall_mesh():
    l, w, h = RECT_LENGTH/2, RECT_WIDTH/2, RECT_HEIGHT/2
    v1f = np.array([-l, -2*h, 0])
    v2f = np.array([+l, -2*h, 0])
    v3f = np.array([+l,     0, 0])
    v4f = np.array([-l,     0, 0])
    v1b, v2b, v3b, v4b = v1f.copy(), v2f.copy(), v3f.copy(), v4f.copy()
    v1b[2]+=w; v2b[2]+=w; v3b[2]+=w; v4b[2]+=w
    pts = np.array([v1f,v2f,v3f,v4f,v1b,v2b,v3b,v4b])
    faces = np.hstack([
        [4,0,1,2,3],[4,4,5,6,7],
        [4,0,1,5,4],[4,1,2,6,5],
        [4,2,3,7,6],[4,3,0,4,7]
    ]).astype(np.int64)
    return pv.PolyData(pts, faces)

plotter = pv.Plotter(window_size=[1400,900])
plotter.set_background("white")

# target 原点
plotter.add_points(np.zeros((1,3)), color="black", point_size=12, label="Target Origin")
arrow_scale = 0.12
plotter.add_arrows(np.array([[0,0,0]]), np.array([[1,0,0]]), mag=arrow_scale, color="red", label="+X")
plotter.add_arrows(np.array([[0,0,0]]), np.array([[0,1,0]]), mag=arrow_scale, color="green", label="+Y")
plotter.add_arrows(np.array([[0,0,0]]), np.array([[0,0,1]]), mag=arrow_scale, color="blue", label="+Z")

# 墙体
plotter.add_mesh(build_wall_mesh(), color="#D95319", opacity=1.0,
                 show_edges=True, edge_color="black", line_width=2, label="Brick Wall")

# Demo 轨迹
plotter.add_mesh(pv.Spline(P_demo, 200), color="pink", line_width=3, label="Demo Traj")
plotter.add_points(P_demo[0], color="green", point_size=14, render_points_as_spheres=True, label="Demo Start")
plotter.add_points(P_demo[-1], color="red", point_size=14, render_points_as_spheres=True, label="Demo End")

# 生成轨迹
plotter.add_mesh(pv.Spline(P_gen, 200), color="royalblue", line_width=3, label="ProMP Gen Traj")
plotter.add_points(P_gen[0], color="green", point_size=14, render_points_as_spheres=True)
plotter.add_points(P_gen[-1], color="red", point_size=14, render_points_as_spheres=True)

plotter.add_legend()
plotter.camera.azimuth=-60
plotter.camera.elevation=25
plotter.camera.zoom(1.3)
plotter.show()
