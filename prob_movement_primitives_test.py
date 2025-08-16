



# """
# Learn a ProMP from a single demo trajectory and generate multiple trajectories
# starting from the same initial point, then visualize in 3D for comparison.

# demo_traj_data: 
#     (70, 4): x_t, y_x, y_y, y_z
#     x_t: canonical time variable in [0, 1]
#     y_x, y_y, y_z: ground-truth x, y, z coordinates
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def rbf_kernel(K, xt, sigma=None):
#     """Compute RBF vector for time xt."""
#     if sigma is None:
#         sigma = 1 / K  # Adjusted for better spread
#     centers = np.linspace(0, 1, K)
#     vec = np.exp(-((xt - centers) ** 2) / (2 * sigma ** 2))
#     vec /= vec.sum() + 1e-10  # Normalize to sum to 1
#     return vec.reshape(K, 1)  # Shape (K, 1)

# def estimate_gaussian_weight(demo_data, K, gamma=1e-6):
#     """
#     Fit ProMP weights for each dimension using normal equation with regularization.
#     Returns mean weights, noise variance, and Phi matrix.
#     """
#     T = demo_data.shape[0]  # Number of frames
#     Phi = np.zeros((T, K), dtype=np.float64)
#     y_x = demo_data[:, 1].reshape(T, 1)  # Shape (T, 1)
#     y_y = demo_data[:, 2].reshape(T, 1)
#     y_z = demo_data[:, 3].reshape(T, 1)
    
#     # Build design matrix Phi
#     for t in range(T):
#         Phi[t, :] = rbf_kernel(K, demo_data[t, 0]).flatten()
    
#     # Fit weights per dimension: omega^i = (Phi^T Phi + gamma I)^(-1) Phi^T y^i
#     omega = np.zeros((3 * K, 1), dtype=np.float64)
#     A = Phi.T @ Phi + gamma * np.eye(K)  # Common for all dimensions
#     for i, y_i in enumerate([y_x, y_y, y_z]):
#         omega[i*K:(i+1)*K] = np.linalg.solve(A, Phi.T @ y_i)
    
#     # Estimate noise variance alpha^2 from residuals
#     residuals = np.zeros((T, 3))
#     for i, y_i in enumerate([y_x, y_y, y_z]):
#         residuals[:, i] = (y_i - Phi @ omega[i*K:(i+1)*K]).flatten()
#     alpha2 = np.sum(residuals ** 2) / (3 * T - 3 * K)  # Pooled variance
    
#     return omega, alpha2, Phi


# def condition_on_start_point(mu_omega, Sigma_omega, start_point, x_start, K, alpha2):
#     """
#     Condition the ProMP to start at start_point at time x_start (usually x_1 = 0).
#     Returns updated mean and covariance of weights.
#     """
#     phi = rbf_kernel(K, x_start)  # Shape (K, 1)
#     Phi_star = np.kron(np.eye(3), phi.T)  # Shape (3, 3K)
    
#     # Joint distribution: [y_start, omega]
#     mu_y_star = Phi_star @ mu_omega  # Expected start point
#     Sigma_yy = Phi_star @ Sigma_omega @ Phi_star.T + alpha2 * np.eye(3)  # Cov of y_start
#     Sigma_yw = Phi_star @ Sigma_omega  # Cross-covariance
#     Sigma_wy = Sigma_yw.T
    
#     # Condition on y_start = start_point
#     diff = start_point.reshape(3, 1) - mu_y_star
#     mu_omega_cond = mu_omega + Sigma_wy @ np.linalg.solve(Sigma_yy, diff)
#     Sigma_omega_cond = Sigma_omega - Sigma_wy @ np.linalg.solve(Sigma_yy, Sigma_yw)
    
#     return mu_omega_cond, Sigma_omega_cond

# def predict_pos_per_axis(xt, K, omega, alpha2, add_noise=True):
#     """Predict 3D position at time xt with optional noise."""
#     phi = rbf_kernel(K, xt)  # Shape (K, 1)
#     y_pred = np.zeros((3, 1), dtype=np.float64)
#     for i in range(3):
#         y_pred[i] = phi.T @ omega[i*K:(i+1)*K]  # Scalar
#     if add_noise:
#         epsilon_y = np.random.multivariate_normal(np.zeros(3), alpha2 * np.eye(3))
#         y_pred += epsilon_y.reshape(3, 1)
#     return y_pred

# def generate_trajectory(T_new, K, mu_omega, Sigma_omega, alpha2):
#     """Generate a new trajectory by sampling weights."""
#     omega_star = np.random.multivariate_normal(mu_omega.flatten(), Sigma_omega)
#     x_new = np.linspace(0, 1, T_new)
#     y_new = np.zeros((T_new, 3))
#     for t in range(T_new):
#         y_new[t, :] = predict_pos_per_axis(x_new[t], K, omega_star, alpha2).flatten()
#     return x_new, y_new

# def plot_3d_trajectories(demo_data, generated_trajs, T_new):
#     """Plot demo and generated trajectories in 3D."""
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Plot demo trajectory
#     ax.plot(demo_data[:, 1], demo_data[:, 2], demo_data[:, 3], 'b-', label='Demo', linewidth=2)
#     ax.scatter(demo_data[0, 1], demo_data[0, 2], demo_data[0, 3], c='b', s=100, marker='o', label='Start (Demo)')
    
#     # Plot generated trajectories
#     colors = ['r', 'g', 'm', 'c']
#     for i, (x_new, y_new) in enumerate(generated_trajs):
#         ax.plot(y_new[:, 0], y_new[:, 1], y_new[:, 2], f'{colors[i % len(colors)]}--', 
#                 label=f'Generated {i+1}', linewidth=1)
#         ax.scatter(y_new[0, 0], y_new[1, 0], y_new[2, 0], c=colors[i % len(colors)], s=50, marker='^')
    
#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.set_zlabel('Z (m)')
#     ax.set_title('3D Trajectory Comparison (Same Start Point)')
#     ax.legend()
#     plt.show()



# if __name__ == "__main__":
#     # Load and validate data
#     demo_traj_data = np.load("demo_keypoint_trajectory.npy")
#     assert demo_traj_data.shape[1] == 4, "Expected shape (T, 4)"
#     T = demo_traj_data.shape[0]  # 70
#     assert np.all((demo_traj_data[:, 0] >= 0) & (demo_traj_data[:, 0] <= 1)), "x_t must be in [0, 1]"
    
#     # Parameters
#     K = 60  # Number of RBFs (reduced to avoid overfitting)
#     gamma = 1e-7  # Regularization for normal equation
#     lambda_ = 1e-7  # Weight covariance scale
#     num_trajs = 1  # Number of generated trajectories
#     T_new = T  # Length of generated trajectories
    
#     # Learn ProMP model
#     mu_omega, alpha2, Phi = estimate_gaussian_weight(demo_traj_data, K, gamma)
#     Sigma_omega = lambda_ * np.eye(3 * K)
    
#     # Condition on demo's start point (y_1 at x_1 = 0)
#     start_point = demo_traj_data[0, 1:4]  # [y_1^x, y_1^y, y_1^z]
#     x_start = demo_traj_data[0, 0]  # Should be 0
#     mu_omega_cond, Sigma_omega_cond = condition_on_start_point(mu_omega, Sigma_omega, start_point, x_start, K, alpha2)
    
#     # Generate multiple trajectories
#     generated_trajs = []
#     for _ in range(num_trajs):
#         x_new, y_new = generate_trajectory(T_new, K, mu_omega_cond, Sigma_omega_cond, alpha2)
#         generated_trajs.append((x_new, y_new))
    
#     # Plot for comparison
#     plot_3d_trajectories(demo_traj_data, generated_trajs, T_new)












"""
Learn a ProMP from a single 3D demo trajectory and condition it to pass through
chosen points (start/end/vias or ALL demo points). Then generate and visualize
trajectories.

Expected input:
  demo_keypoint_trajectory.npy  # shape (T,4): [x_t in [0,1], y_x, y_y, y_z]
"""

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# Basis functions and design
# ------------------------------
def rbf_kernel(K, xt, sigma=None):
    """
    Gaussian RBF row vector (K,) at canonical time xt in [0,1].
    Centers are evenly spaced; rows are normalized to sum to 1.
    """
    if sigma is None:
        sigma = 1.0 / K  # reasonable default in canonical time
    centers = np.linspace(0.0, 1.0, K)
    v = np.exp(-0.5 * ((xt - centers) ** 2) / (sigma ** 2))
    s = v.sum()
    if s > 1e-12:
        v = v / s
    return v  # shape (K,)

def build_design_matrix(xs, K, sigma=None):
    """
    Build Phi \in R^{T x K} by stacking rbf_kernel rows for each xt in xs.
    """
    T = len(xs)
    Phi = np.zeros((T, K), dtype=np.float64)
    for t, xt in enumerate(xs):
        Phi[t, :] = rbf_kernel(K, xt, sigma)
    return Phi  # (T, K)

def build_block_phi(xt, K, D=3, sigma=None):
    """
    Build block feature Phi_D(x) = I_D \otimes phi(x)^T \in R^{D x (KD)}.
    """
    phi = rbf_kernel(K, xt, sigma).reshape(1, K)  # (1,K)
    return np.kron(np.eye(D), phi)  # (D, KD)

def build_Phi_constraints(xs_constraints, K, D=3, sigma=None):
    """
    Stack block feature rows for multiple constraints.
    Returns Phi_c \in R^{(M*D) x (K*D)}.
    """
    blocks = [build_block_phi(xc, K, D, sigma) for xc in xs_constraints]
    return np.vstack(blocks)  # (M*D, K*D)



def fit_weights_single_demo(demo_traj, K, sigma=None, ridge=1e-6):
    """
    Fit weights per axis using ridge least squares:
      w^d = (Phi^T Phi + ridge I)^{-1} Phi^T y^d
    Returns:
      mu_w : (K*D, 1) stacked weights [w^x; w^y; w^z]
      alpha2 : scalar observation noise variance (pooled across axes)
      Phi : (T,K) design matrix for the demo xs
    """
    T = demo_traj.shape[0]
    xs = demo_traj[:, 0]
    Y  = demo_traj[:, 1:4]               # (T,3)
    D  = Y.shape[1]

    Phi = build_design_matrix(xs, K, sigma)   # (T,K)
    A   = Phi.T @ Phi + ridge * np.eye(K)     # (K,K)

    mu_w = np.zeros((K*D, 1), dtype=np.float64)

    # Solve per axis, stack
    for d in range(D):
        y_d = Y[:, d].reshape(T, 1)                       # (T,1)
        w_d = np.linalg.solve(A, Phi.T @ y_d)             # (K,1)
        mu_w[d*K:(d+1)*K, :] = w_d

    # Residuals and pooled variance (OLS DoF approx; fine for single demo)
    residuals = np.zeros((T, D))
    for d in range(D):
        w_d = mu_w[d*K:(d+1)*K, :]                        # (K,1)
        residuals[:, d] = (Y[:, d].reshape(T,1) - Phi @ w_d).ravel()
    dof = max(1, D*T - D*K)  # guard
    alpha2 = float(np.sum(residuals**2) / dof)

    return mu_w, alpha2, Phi



def condition_on_points(mu_w, lambda2, xs_constraints, y_constraints, K, D=3, sigma=None, sigma_c2=1e-4):
    """
    ProMP conditioning with prior Sigma_w = lambda2 * I (size K*D),
    and constraint noise R = sigma_c2 * I (size M*D).

    Inputs:
      mu_w           : (K*D,1) mean weights from single-demo fit
      lambda2        : scalar (prior covariance scale)
      xs_constraints : list/array of M canonical times in [0,1]
      y_constraints  : (M,D) target positions
      K, D, sigma    : basis params
      sigma_c2       : scalar constraint noise

    Returns:
      mu_w_cond      : (K*D,1)
      Sigma_w_cond   : (K*D,K*D)
    """
    xs_constraints = np.asarray(xs_constraints).ravel()
    M = xs_constraints.shape[0]
    assert y_constraints.shape == (M, D)

    Phi_c = build_Phi_constraints(xs_constraints, K, D, sigma)     # (M*D, K*D)
    Y_c   = y_constraints.reshape(M*D, 1)                          # (M*D, 1)

    Sigma_w = lambda2 * np.eye(K*D)                                # (K*D, K*D)
    R       = sigma_c2 * np.eye(M*D)                               # (M*D, M*D)

    S = Phi_c @ Sigma_w @ Phi_c.T + R                              # (M*D, M*D)
    K_gain = Sigma_w @ Phi_c.T @ np.linalg.solve(S, np.eye(M*D))   # (K*D, M*D)

    mu_w_cond    = mu_w + K_gain @ (Y_c - Phi_c @ mu_w)            # (K*D, 1)
    Sigma_w_cond = Sigma_w - K_gain @ Phi_c @ Sigma_w              # (K*D, K*D)

    return mu_w_cond, Sigma_w_cond

# ------------------------------
# Rollout / generation
# ------------------------------
def predict_mean_at(xs, K, mu_w, D=3, sigma=None):
    """
    Mean trajectory y_hat(xs) using mean weights mu_w.
    Returns (len(xs), D)
    """
    xs = np.asarray(xs)
    Tn = xs.shape[0]
    Y  = np.zeros((Tn, D), dtype=np.float64)
    for i, xt in enumerate(xs):
        Phi_D = build_block_phi(xt, K, D, sigma)  # (D, K*D)
        Y[i, :] = (Phi_D @ mu_w).ravel()
    return Y

def sample_trajectory(xs, K, mu_w, Sigma_w, alpha2=0.0, D=3, sigma=None, rng=None):
    """
    Sample a full trajectory by sampling weights from N(mu_w, Sigma_w)
    and (optionally) adding observation noise alpha2 at each timestep.
    """
    if rng is None:
        rng = np.random.default_rng()
    w_s = rng.multivariate_normal(mu_w.ravel(), Sigma_w).reshape(-1,1)  # (K*D,1)
    Y   = predict_mean_at(xs, K, w_s, D, sigma)                         # (Tn,D)
    if alpha2 > 0.0:
        noise = rng.multivariate_normal(np.zeros(D), alpha2*np.eye(D), size=len(xs))
        Y = Y + noise
    return Y

# ------------------------------
# Visualization
# ------------------------------
def plot_3d(demo, curves, labels, title="Trajectories"):
    fig = plt.figure(figsize=(9,7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(demo[:,0], demo[:,1], demo[:,2], 'b-', lw=2, label='Demo')
    ax.scatter(demo[0,0], demo[0,1], demo[0,2], c='b', s=60, marker='o', label='Demo start')

    colors = ['r','g','m','c','y','k']
    for i, Y in enumerate(curves):
        c = colors[i % len(colors)]
        ax.plot(Y[:,0], Y[:,1], Y[:,2], c=c, ls='--', lw=1.5, label=labels[i])
        ax.scatter(Y[0,0], Y[0,1], Y[0,2], c=c, s=50, marker='^')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    # keep aspect reasonable
    def rng(v): 
        return max(1e-9, np.ptp(v))
    ax.set_box_aspect((rng(demo[:,0]), rng(demo[:,1]), rng(demo[:,2])))
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    # Load demo
    demo_traj = np.load("demo_keypoint_trajectory.npy")  # (T,4): [x_t, x, y, z]
    assert demo_traj.shape[1] == 4, "demo_keypoint_trajectory.npy must be shape (T,4)"

    xs = demo_traj[:,0]             # (T,)
    Yd = demo_traj[:,1:4]           # (T,3)
    T  = len(xs)
    D  = 3

    # --- Hyperparameters ---
    K        = 30          # # of RBFs (try 15-30)
    sigma    = 1 / K        # RBF width in canonical time (try 0.05-0.12)
    ridge    = 1e-6        # ridge for LS
    lambda2  = 1e-7        # prior weight covariance scale (Sigma_w = lambda2 * I)
    sigma_c2 = 1e-6        # constraint noise (smaller -> harder)
    n_samples= 1           # # of sampled trajectories to draw
    use_all_demo_points = False  # True: condition on ALL demo points; False: subset

    # --- Learn from single demo ---
    mu_w, alpha2, Phi = fit_weights_single_demo(demo_traj, K, sigma, ridge)  # mu_w: (K*D,1)

    # --- Build constraints ---
    if use_all_demo_points:
        xs_c = xs.copy()                       # all demo times
        ys_c = Yd.copy()                       # all demo positions
    else:
        # Example: start, 3 via points, end
        idxs = [0, max(1,T//4), max(2,T//2), max(3,3*T//4), T-1]
        xs_c = xs[idxs]
        ys_c = Yd[idxs, :]

    # --- Condition weights ---
    mu_w_cond, Sigma_w_cond = condition_on_points(mu_w, lambda2, xs_c, ys_c, K, D, sigma, sigma_c2)

    # --- Rollout mean and samples ---
    y_mean = predict_mean_at(xs, K, mu_w_cond, D, sigma)  # (T,3)
    curves = [y_mean]
    labels = ["Conditioned mean"]

    rng = np.random.default_rng(0)
    for i in range(n_samples):
        y_sample = sample_trajectory(xs, K, mu_w_cond, Sigma_w_cond, alpha2=alpha2, D=D, sigma=sigma, rng=rng)
        curves.append(y_sample)
        labels.append(f"Sample {i+1}")

    # --- Plot ---
    plot_3d(Yd, curves, labels, title="ProMP: conditioned reproduction & samples")
