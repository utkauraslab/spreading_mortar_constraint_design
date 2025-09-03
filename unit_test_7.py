
"""
    Validate & visualize the trowel pose trajectory in the brick-wall (target) local frame.
    tool's pose trajectory in the brick-wall (target) local frame is obtained by tool_target_transform.py script.
"""


import numpy as np
import pyvista as pv


POSE_PATH = "trowel_pose_in_target_frame.npy"
T_seq = np.load(POSE_PATH)
assert T_seq.ndim == 3 and T_seq.shape[1:] == (4,4), "expect (N,4,4) pose sequence in target frame"

TRIANGLE_EDGE = 0.15      
TRIANGLE_WIDTH = 0.08     
TRIAD_STRIDE   = 8        
PATH_SAMPLES   = 800      

RECT_LENGTH = 0.85   
RECT_HEIGHT = 0.04   
RECT_WIDTH  = 0.16   

def add_canonical_triangle_at_pose(plotter, T, length=TRIANGLE_EDGE, width=TRIANGLE_WIDTH,
                                   color="gray", opacity=0.5, edge=True):
    """
      canonical triangle:
      v1 = (-L/2, 0), v2 = (L/2, -W/2), v3 = (L/2, +W/2)
    """
    Rm = T[:3, :3]; p = T[:3, 3]
    x = Rm[:, 0]; y = Rm[:, 1]  

    v1 = p + (-length/2)*x + 0.0*y
    v2 = p + ( length/2)*x + (-width/2)*y
    v3 = p + ( length/2)*x + ( width/2)*y

    tri = np.stack([v1, v2, v3], axis=0)
    face = np.hstack([3, 0, 1, 2]).astype(np.int64)
    plotter.add_mesh(pv.PolyData(tri, face), color=color, opacity=opacity,
                     show_edges=edge, edge_color="black")

def build_wall_mesh():
    
    l, w, h = RECT_LENGTH/2, RECT_WIDTH/2, RECT_HEIGHT/2
    
    v1f = np.array([-l, -2*h, 0])
    v2f = np.array([+l, -2*h, 0])
    v3f = np.array([+l,     0, 0])
    v4f = np.array([-l,     0, 0])
    
    v1b, v2b, v3b, v4b = v1f.copy(), v2f.copy(), v3f.copy(), v4f.copy()
    v1b[2] += w; v2b[2] += w; v3b[2] += w; v4b[2] += w
    pts = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
    faces = np.hstack([
        [4, 0,1,2,3],  # front
        [4, 4,5,6,7],  # back
        [4, 0,1,5,4],
        [4, 1,2,6,5],
        [4, 2,3,7,6],
        [4, 3,0,4,7],
    ]).astype(np.int64)
    return pv.PolyData(pts, faces)

plotter = pv.Plotter(window_size=[1400, 900])
plotter.set_background("white")

plotter.add_points(np.zeros((1,3)), color="black", point_size=12, label="Target Origin")
arrow_scale = 0.12
plotter.add_arrows(np.array([[0,0,0]]), np.array([[ 1, 0, 0]]), mag=arrow_scale, color="red",   label="+X (right)")
plotter.add_arrows(np.array([[0,0,0]]), np.array([[ 0,1, 0]]), mag=arrow_scale, color="green", label="+Y (down)")
plotter.add_arrows(np.array([[0,0,0]]), np.array([[ 0, 0,1]]), mag=arrow_scale, color="blue",  label="+Z (into)")


plotter.add_mesh(build_wall_mesh(), color="#D95319", opacity=1.0,
                 show_edges=True, edge_color="black", line_width=2, label="Brick Wall")

P = T_seq[:, :3, 3]
plotter.add_mesh(pv.Spline(P, PATH_SAMPLES), color="#d62728", line_width=5, label="Path")
plotter.add_points(P[0],  color="green", point_size=16, render_points_as_spheres=True, label="Start")
plotter.add_points(P[-1], color="red",   point_size=16, render_points_as_spheres=True, label="End")



for i in range(len(T_seq)):
    add_canonical_triangle_at_pose(plotter, T_seq[i], color="gray", opacity=0.45)

plotter.add_legend()
plotter.camera.azimuth = -60
plotter.camera.elevation = 25
plotter.camera.zoom(1.3)
plotter.show()
