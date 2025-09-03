# import os
# import numpy as np
# import pyvista as pv

# # ========= 配置 =========
# GEN_PATH = "generated_pose_se3_K4.npy"  # 你的生成结果文件
# TRIANGLE_EDGE_SIZE = 0.15               # 抹子三角形的“长度”（沿局部 x）
# TRIANGLE_WIDTH     = 0.08               # 抹子三角形的“宽度”（沿局部 y）
# TRIAD_STRIDE       = 8                  # 每隔多少帧画一个三角形，避免太密
# APPLY_VIZ_FLIP     = True               # 是否把(y, z)翻到“y向上 / z出纸面”的人眼直观系

# # 砖墙体（可视化尺度，和你之前一致）
# RECTANGLE_LENGTH = 0.85   # 沿 x
# RECTANGLE_HEIGHT = 0.04   # 沿 y（会画成从 y=-2h 到 y=0 的带）
# RECTANGLE_WIDTH  = 0.16   # 沿 z（墙厚）

# # ========= 读取生成的 pose =========
# T_gen = np.load(GEN_PATH)              # (M,4,4), in wall local frame: x-right, y-down, z-into-page
# assert T_gen.ndim == 3 and T_gen.shape[1:] == (4,4)

# # 仅用于“可视化”的坐标翻转（把 y 从 down -> up，把 z 从 into -> out）
# if APPLY_VIZ_FLIP:
#     S = np.eye(4)
#     S[:3, :3] = np.diag([1, -1, -1])  # 视觉用：x 保持，y/z 取反
#     T_viz = (S[None, ...] @ T_gen)
# else:
#     T_viz = T_gen.copy()

# # ========= 构建砖墙体（与 T_viz 同一坐标系）=========
# l, w, h = RECTANGLE_LENGTH/2, RECTANGLE_WIDTH/2, RECTANGLE_HEIGHT/2
# # 前面 (z=0) 从 y=-2h 到 y=0 的“抹灰带”
# v1f = np.array([-l, -2*h, 0])
# v2f = np.array([+l, -2*h, 0])
# v3f = np.array([+l,     0, 0])
# v4f = np.array([-l,     0, 0])
# # 背面 (z=+w)
# v1b, v2b, v3b, v4b = v1f.copy(), v2f.copy(), v3f.copy(), v4f.copy()
# v1b[2] += w; v2b[2] += w; v3b[2] += w; v4b[2] += w

# pts_w = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
# faces = np.hstack([
#     [4, 0, 1, 2, 3],   # front
#     [4, 4, 5, 6, 7],   # back
#     [4, 0, 1, 5, 4],
#     [4, 1, 2, 6, 5],
#     [4, 2, 3, 7, 6],
#     [4, 3, 0, 4, 7],
# ]).astype(np.int64)
# wall_mesh = pv.PolyData(pts_w, faces)

# # ========= 创建 Plotter =========
# plotter = pv.Plotter(window_size=[1400, 900])
# plotter.set_background("white")

# # 墙坐标轴（可选：为“x右 y上 z出纸面”的视觉标注）
# plotter.add_points(np.zeros((1, 3)), color="black", point_size=12, label="Wall Origin")
# arrow_scale = 0.12
# plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[1, 0, 0]]),  # +X
#                    mag=arrow_scale, color="red",   label="+X")
# plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[0, 1, 0]]),  # +Y
#                    mag=arrow_scale, color="green", label="+Y")
# plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[0, 0, 1]]),  # +Z
#                    mag=arrow_scale, color="blue",  label="+Z")

# # 墙体
# plotter.add_mesh(
#     wall_mesh, color="#D95319", opacity=1.0,
#     show_edges=True, edge_color="black", line_width=2,
#     label="Wall Volume"
# )

# # ========= 画生成轨迹的“位置曲线” =========
# P = T_viz[:, :3, 3]
# plotter.add_mesh(pv.Spline(P, 1000), color="#d62728", line_width=5, label="Generated Path")
# plotter.add_points(P[0],  color="green", point_size=16, render_points_as_spheres=True, label="Start")
# plotter.add_points(P[-1], color="red",   point_size=16, render_points_as_spheres=True, label="End")

# # ========= 在每一帧的 pose 上画“canonical triangle” =========
# def add_canonical_triangle_at_pose(T, length=TRIANGLE_EDGE_SIZE, width=TRIANGLE_WIDTH, color="gray", opacity=0.5):
#     """
#     在 4x4 pose T 上画一个位于局部 x–y 平面的三角形：
#       顶点（局部坐标）：
#         v1 = [-length/2, 0]
#         v2 = [ +length/2, -width/2]
#         v3 = [ +length/2, +width/2]
#     """
#     Rm = T[:3, :3]
#     t  = T[:3, 3]
#     x = Rm[:, 0]  # 局部 x 轴（列向量）
#     y = Rm[:, 1]  # 局部 y 轴

#     v1_local = np.array([-length/2, 0.0])
#     v2_local = np.array([ +length/2, -width/2])
#     v3_local = np.array([ +length/2, +width/2])

#     v1 = t + v1_local[0]*x + v1_local[1]*y
#     v2 = t + v2_local[0]*x + v2_local[1]*y
#     v3 = t + v3_local[0]*x + v3_local[1]*y

#     tri = np.stack([v1, v2, v3], axis=0)
#     face = np.hstack([3, 0, 1, 2]).astype(np.int64)
#     tri_mesh = pv.PolyData(tri, face)
#     plotter.add_mesh(tri_mesh, color=color, opacity=opacity, show_edges=True)

# # 稀疏绘制（避免太密）
# M = len(T_viz)
# for i in range(0, M, max(1, TRIAD_STRIDE)):
#     add_canonical_triangle_at_pose(T_viz[i], color="gray", opacity=0.45)

# # ========= 相机视角 =========
# plotter.add_legend()
# plotter.camera.azimuth = -60
# plotter.camera.elevation = 25
# plotter.camera.zoom(1.3)
# plotter.show()




import numpy as np
import pyvista as pv


#GEN_PATH = "generated_pose_se3_K4.npy"  

GEN_PATH = "lpvds_reproduction.npy"
T_gen = np.load(GEN_PATH)
assert T_gen.ndim == 3 and T_gen.shape[1:] == (4,4)


TRIANGLE_EDGE_SIZE = 0.15    
TRIANGLE_WIDTH     = 0.08    
TRIAD_STRIDE       = 8       
PATH_SAMPLES       = 1000   



RECTANGLE_LENGTH = 0.85  
RECTANGLE_HEIGHT = 0.04   
RECTANGLE_WIDTH  = 0.16   


l, w, h = RECTANGLE_LENGTH/2, RECTANGLE_WIDTH/2, RECTANGLE_HEIGHT/2
v1f = np.array([-l, -2*h, 0])
v2f = np.array([+l, -2*h, 0])
v3f = np.array([+l,     0, 0])
v4f = np.array([-l,     0, 0])
v1b, v2b, v3b, v4b = v1f.copy(), v2f.copy(), v3f.copy(), v4f.copy()
v1b[2] += w; v2b[2] += w; v3b[2] += w; v4b[2] += w

pts_w = np.array([v1f, v2f, v3f, v4f, v1b, v2b, v3b, v4b])
faces = np.hstack([
    [4, 0, 1, 2, 3],   # front
    [4, 4, 5, 6, 7],   # back
    [4, 0, 1, 5, 4],
    [4, 1, 2, 6, 5],
    [4, 2, 3, 7, 6],
    [4, 3, 0, 4, 7],
]).astype(np.int64)
wall_mesh = pv.PolyData(pts_w, faces)

# ========= Plotter =========
plotter = pv.Plotter(window_size=[1400, 900])
plotter.set_background("white")

#（target local frame：x-right, y-down, z-into-page）
plotter.add_points(np.zeros((1, 3)), color="black", point_size=12, label="Target Origin")
arrow_scale = 0.12
plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[ 1, 0, 0]]), mag=arrow_scale, color="red",   label="+X (right)")
plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[ 0,1, 0]]), mag=arrow_scale, color="green", label="+Y (down)")
plotter.add_arrows(cent=np.array([[0, 0, 0]]), direction=np.array([[ 0, 0,1]]), mag=arrow_scale, color="blue",  label="+Z (into)")


plotter.add_mesh(wall_mesh, color="#D95319", opacity=1.0,
                 show_edges=True, edge_color="black", line_width=2,
                 label="Brick Wall")



# ========= 生成轨迹（位置 + triangle）=========
P = T_gen[:, :3, 3]
plotter.add_mesh(pv.Spline(P, PATH_SAMPLES), color="#d62728", line_width=5, label="Generated Path")
plotter.add_points(P[0],  color="green", point_size=16, render_points_as_spheres=True, label="Start")
plotter.add_points(P[-1], color="red",   point_size=16, render_points_as_spheres=True, label="End")

def add_canonical_triangle_at_pose(T, length=TRIANGLE_EDGE_SIZE, width=TRIANGLE_WIDTH, color="gray", opacity=0.5):
    
    Rm = T[:3, :3]
    t = T[:3, 3]
    # t[0] = -t[0]
    x = Rm[:, 0]         
    y = Rm[:, 1]         
    v1_local = np.array([-length/2, 0.0])
    v2_local = np.array([ +length/2, -width/2])
    v3_local = np.array([ +length/2, +width/2])
    v1 = t + v1_local[0]*x + v1_local[1]*y
    v2 = t + v2_local[0]*x + v2_local[1]*y
    v3 = t + v3_local[0]*x + v3_local[1]*y
    tri  = np.stack([v1, v2, v3], axis=0)
    face = np.hstack([3, 0, 1, 2]).astype(np.int64)
    plotter.add_mesh(pv.PolyData(tri, face), color=color, opacity=opacity, show_edges=True)

M = len(T_gen)
print(M)
for i in range(0, 100, max(1, TRIAD_STRIDE)):
    add_canonical_triangle_at_pose(T_gen[i], color="gray", opacity=0.45)


plotter.add_legend()
plotter.camera.azimuth = -60
plotter.camera.elevation = 25
plotter.camera.zoom(1.3)
plotter.show()
