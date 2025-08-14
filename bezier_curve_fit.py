

#!/usr/bin/env python3
# Fit & viz a piecewise quadratic Bézier spline to a 3D trajectory.
# Replace `points` with your real (N,3) array.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 registers 3D projection

from pathlib import Path

# ---------------------- Utilities ----------------------

def arc_length_param(points: np.ndarray) -> np.ndarray:
    """Return normalized cumulative arc-length parameter s in [0,1]."""
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s / (s[-1] if s[-1] > 0 else 1.0)

def bezier_eval(P0, P1, P2, tau: np.ndarray) -> np.ndarray:
    """Evaluate quadratic Bézier at local tau in [0,1]."""
    tau = tau[:, None]
    return ((1 - tau)**2) * P0 + 2*(1 - tau)*tau * P1 + (tau**2) * P2

def fit_quadratic_bezier_segment(Pseg: np.ndarray, tau: np.ndarray,
                                 P0=None, P2=None):
    """
    Closed-form least-squares for the middle control point P1 of a quadratic
    Bézier with endpoints fixed to first/last sample of the segment.
    Works in any dimension (here 3D).
    """
    if P0 is None: P0 = Pseg[0]
    if P2 is None: P2 = Pseg[-1]
    b0 = (1 - tau)**2
    b1 = 2 * (1 - tau) * tau
    b2 = tau**2
    num = (b1[:, None] * (Pseg - (b0[:, None] * P0 + b2[:, None] * P2))).sum(axis=0)
    den = (b1**2).sum()
    P1 = num / max(den, 1e-12)
    return P0, P1, P2

def segment_error(Pseg, P0, P1, P2, tau):
    Q = bezier_eval(P0, P1, P2, tau)
    err = np.linalg.norm(Q - Pseg, axis=1)
    return err.max(), err

def adaptive_fit(points: np.ndarray, tol: float, min_pts: int = 6):
    """
    Top-down adaptive splitting:
      - try a single quadratic on the range,
      - if max error > tol, split at worst point and recurse.
    Returns: segments [(i0,i1), ...], controls [[P0,P1,P2], ...]
    """
    s = arc_length_param(points)
    segments = []
    stack = [(0, len(points)-1)]

    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 + 1 < min_pts:
            segments.append((i0, i1))
            continue

        s0, s1 = s[i0], s[i1]
        tau = (s[i0:i1+1] - s0) / max(s1 - s0, 1e-12)
        P0, P1, P2 = fit_quadratic_bezier_segment(points[i0:i1+1], tau, points[i0], points[i1])
        emax, e = segment_error(points[i0:i1+1], P0, P1, P2, tau)

        if emax <= tol:
            segments.append((i0, i1))
        else:
            j_local = int(np.argmax(e))
            j = int(np.clip(i0 + j_local, i0 + 3, i1 - 3))  # keep both sides viable
            stack.append((i0, j))
            stack.append((j, i1))

    segments.sort(key=lambda ab: ab[0])

    # Build controls per segment
    ctrls = []
    for (i0, i1) in segments:
        s0, s1 = s[i0], s[i1]
        tau = (s[i0:i1+1] - s0) / max(s1 - s0, 1e-12)
        P0, P1, P2 = fit_quadratic_bezier_segment(points[i0:i1+1], tau, points[i0], points[i1])
        ctrls.append([P0.copy(), P1.copy(), P2.copy()])

    # Optional: enforce simple C1 smoothing at joins
    for k in range(len(ctrls)-1):
        P0, P1, P2 = ctrls[k]
        Q0, Q1, Q2 = ctrls[k+1]
        Q0[:] = P2
        v_left  = P2 - P1
        v_right = Q1 - Q0
        v = 0.5 * (v_left + v_right)
        P1[:] = P2 - v
        Q1[:] = Q0 + v

    return segments, ctrls

def evaluate_spline(ctrls, samples_per_segment: int = 100) -> np.ndarray:
    """Sample the piecewise quadratic spline densely for visualization."""
    pts = []
    for (P0, P1, P2) in ctrls:
        tau = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        pts.append(bezier_eval(P0, P1, P2, tau))
    if ctrls:
        pts.append(bezier_eval(*ctrls[-1], np.array([1.0])))
    return np.vstack(pts) if pts else np.zeros((0, 3))

def compute_sample_errors(points, segments, ctrls) -> np.ndarray:
    """Per-sample Euclidean error against its segment fit."""
    s = arc_length_param(points)
    errs = np.zeros(len(points))
    for seg_idx, (i0, i1) in enumerate(segments):
        s0, s1 = s[i0], s[i1]
        tau = (s[i0:i1+1] - s0) / max(s1 - s0, 1e-12)
        P0, P1, P2 = ctrls[seg_idx]
        Q = bezier_eval(P0, P1, P2, tau)
        errs[i0:i1+1] = np.linalg.norm(Q - points[i0:i1+1], axis=1)
    return errs




if __name__ == "__main__":
    arr = np.load("demo_keypoint_trajectory.npy")  # shape (N,4)
    
    t_can = arr[:, 0].astype(float)
    points = arr[:, 1:4].astype(float)  # (N,3)

    tol = 0.05  # tolerance in the same units as your points (e.g., meters)
    segments, ctrls = adaptive_fit(points, tol=tol)
    curve_pts = evaluate_spline(ctrls, samples_per_segment=100)
    errs = compute_sample_errors(points, segments, ctrls)

    # ---- 3D plot: original vs fitted spline ----
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o', linestyle='',
            label='Original samples')
    ax.plot(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2],
            label='Quadratic Bézier spline (fit)')
    # draw control polygons
    for (P0, P1, P2) in ctrls:
        ax.plot([P0[0], P1[0], P2[0]],
                [P0[1], P1[1], P2[1]],
                [P0[2], P1[2], P2[2]], linestyle='--')
    ax.set_title(f"3D Trajectory vs. Quadratic Bézier Spline | Segments: {len(ctrls)} | Tol={tol}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ---- Error plot ----
    plt.figure(figsize=(7, 4))
    plt.plot(errs, marker='o', linestyle='-', label='Per-sample error')
    plt.axhline(tol, linestyle='--', label='Tolerance')
    plt.title("Fitting Error per Sample")
    plt.xlabel("Sample index")
    plt.ylabel("Euclidean error")
    plt.legend()
    plt.tight_layout()
    plt.show()
    