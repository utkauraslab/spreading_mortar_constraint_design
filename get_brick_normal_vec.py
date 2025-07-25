# import cv2

# def click_event(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(f"Clicked at: x={x}, y={y}")

# # Load your image
# img = cv2.imread('./bricklaying_data/frame_0067.png')
# cv2.imshow('Image', img)
# cv2.setMouseCallback('Image', click_event)

# cv2.waitKey(0)
# cv2.destroyAllWindows()






# # vis_tip_and_brick_normals.py
# import os, cv2, torch, numpy as np, matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from depth_anything.depth_anything_v2.dpt import DepthAnythingV2


# brick_px = np.array([[1416, 1096],   # u,v  (frame_0067.png)
#                      [1792, 1064],
#                      [1925, 1180]])  


# fx = fy = 836.0
# cx, cy = 979.0, 632.0


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# cfg = dict(encoder='vitl', features=256,
#            out_channels=[256,512,1024,1024])
# depth_net = DepthAnythingV2(**cfg).to(device).eval()
# ckpt = 'depth_anything/checkpoints/depth_anything_v2_vitl.pth'
# depth_net.load_state_dict(torch.load(ckpt, map_location=device))

# img = cv2.imread("bricklaying_data/frame_0067.png")
# if img is None:
#     raise FileNotFoundError("frame_0067.png not found")
# inv = depth_net.infer_image(img)                  # inverse depth
# rel = inv.max() - inv                             


# scale = 0.2 / rel[int(cy), int(cx)]
# Zmap  = rel * scale                               # 米单位


# P_cam = []
# for u,v in brick_px:
#     Z = Zmap[int(v), int(u)]
#     X = (u - cx) * Z / fx
#     Y = (v - cy) * Z / fy
#     P_cam.append([X,Y,Z])
# P_cam = np.array(P_cam)                           # (3,3)


# a, b = P_cam[1]-P_cam[0], P_cam[2]-P_cam[0]
# brick_n = np.cross(a, b)
# brick_n /= np.linalg.norm(brick_n)
# brick_n = -brick_n
# print("brick normal =", brick_n)


# pts = torch.load("trowel_tip_vertex_trajectories_3d.pt")  # (3,T,3)
# pts = pts.numpy() if isinstance(pts, torch.Tensor) else pts
# N, T, _ = pts.shape


# # tri_n = np.cross(pts[1]-pts[0], pts[2]-pts[0], ax=1)

# a = pts[1] - pts[0]          # (T,3)
# b = pts[2] - pts[0]          # (T,3)
# tri_n = np.cross(a, b)       # (T,3)
# tri_n = tri_n / np.linalg.norm(tri_n, axis=1, keepdims=True)
# centers = pts.mean(0)                             # (T,3)


# fig = plt.figure(figsize=(6,5))
# ax  = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# ax.set_box_aspect([1,1,1])

# def draw(k):
#     ax.cla()
#     # 绿色三角
#     tri = Poly3DCollection([pts[:,k]], alpha=0.3, facecolor='limegreen')
#     ax.add_collection3d(tri)

#     # red - trowel
#     sc = np.linalg.norm(pts[0,k]-pts[1,k])*0.6
#     ax.quiver(*centers[k], *(tri_n[k]*sc), color='red', linewidth=2)

#     # blue - brick
#     ax.quiver(*centers[k], *(brick_n*sc), color='blue', linewidth=2)

#     # ------------ Camera XYZ 轴 ------------
#     axis_len = sc*1.5
#     ax.quiver(0,0,0, axis_len,0,0, color='r', linewidth=2)
#     ax.quiver(0,0,0, 0,axis_len,0, color='g', linewidth=2)
#     ax.quiver(0,0,0, 0,0,axis_len, color='b', linewidth=2)
#     ax.text(axis_len,0,0,'X',color='r')
#     ax.text(0,axis_len,0,'Y',color='g')
#     ax.text(0,0,axis_len,'Z',color='b')
#     # ---------------------------------------

#     ax.set_title(f"Frame {k}/{T-1}")
#     allp = pts.reshape(-1,3); pad=(allp.max(0)-allp.min(0))*0.1
#     ax.set_xlim(allp[:,0].min()-pad[0], allp[:,0].max()+pad[0])
#     ax.set_ylim(allp[:,1].min()-pad[1], allp[:,1].max()+pad[1])
#     ax.set_zlim(allp[:,2].min()-pad[2], allp[:,2].max()+pad[2])
#     ax.set_xlabel('X (right)'); ax.set_ylabel('Y (down)'); ax.set_zlabel('Z (forward)')
#     ax.set_box_aspect([1,1,1])
#     ax.text2D(0.02,0.95,"red =trowel n   blue =brick n", transform=ax.transAxes)

# idx=[0]
# def on_key(e):
#     if e.key in ('right','left'):
#         idx[0] = (idx[0]+1)%T if e.key=='right' else (idx[0]-1)%T
#         draw(idx[0])
#     elif e.key=='escape':
#         plt.close(fig)

# fig.canvas.mpl_connect('key_press_event', on_key)
# draw(0)

# plt.show()









# # compute_normal_angles.py
# import os, cv2, torch, numpy as np
# from depth_anything.depth_anything_v2.dpt import DepthAnythingV2


# fx = fy = 836.0
# cx, cy = 979.0, 632.0


# brick_px = np.array([[1416, 1096],
#                      [1792, 1064],
#                      [1925, 1180]], int)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# net = DepthAnythingV2(encoder='vitl', features=256,
#                       out_channels=[256,512,1024,1024]).to(device).eval()
# net.load_state_dict(torch.load(
#     'depth_anything/checkpoints/depth_anything_v2_vitl.pth', map_location=device))

# img = cv2.imread("bricklaying_data/frame_0067.png")
# inv = net.infer_image(img)
# rel = inv.max() - inv
# scale = 0.2 / rel[int(cy), int(cx)]           
# Zmap  = rel * scale


# P_cam = []
# for u,v in brick_px:
#     Z = Zmap[v,u]
#     X = (u-cx)*Z/fx; Y = (v-cy)*Z/fy
#     P_cam.append([X,Y,Z])
# P_cam = np.array(P_cam)


# # a,b = P_cam[1]-P_cam[0], P_cam[2]-P_cam[0]
# # brick_n = -np.cross(a,b)
# # brick_n /= np.linalg.norm(brick_n)
# # brick_n = -brick_n


# a, b = P_cam[1]-P_cam[0], P_cam[2]-P_cam[0]
# brick_n = np.cross(a, b)
# brick_n /= np.linalg.norm(brick_n)
# brick_n = -brick_n
# print("brick normal =", brick_n)
# # pts = torch.load("trowel_tip_vertex_trajectories_3d.pt").numpy()   # (3,T,3)

# data = torch.load("trowel_tip_vertex_trajectories_3d.pt")   
# pts  = data.cpu().numpy() if isinstance(data, torch.Tensor) else data   
# a = pts[1]-pts[0]; b = pts[2]-pts[0]
# tri_n = np.cross(a, b)                           # (T,3)
# tri_n /= np.linalg.norm(tri_n, axis=1, keepdims=True)


# dot = (tri_n @ brick_n)                          
# dot = np.clip(dot, -1.0, 1.0)                    
# angles_deg = np.degrees(np.arccos(dot))          


# print(angles_deg)
# # print(f"  min  : {angles_deg.min():.2f}°")
# # print(f"  mean : {angles_deg.mean():.2f}°")
# # print(f"  max  : {angles_deg.max():.2f}°")

# # np.save("normal_angle_per_frame.npy", angles_deg)
# # print("Saved per-frame angles to normal_angle_per_frame.npy")









# vis_tip_and_brick_normals.py
import os, cv2, torch, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2


# -------------------- Brick Points & Camera Params --------------------
brick_px = np.array([[1416, 1096], [1792, 1064], [1925, 1180]])  # u,v
fx = fy = 836.0
cx, cy = 979.0, 632.0

# -------------------- Load Depth Model --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = dict(encoder='vitl', features=256, out_channels=[256,512,1024,1024])
depth_net = DepthAnythingV2(**cfg).to(device).eval()
ckpt = 'depth_anything/checkpoints/depth_anything_v2_vitl.pth'
depth_net.load_state_dict(torch.load(ckpt, map_location=device))

# -------------------- Load Image and Depth --------------------
img = cv2.imread("bricklaying_data/frame_0067.png")
if img is None:
    raise FileNotFoundError("frame_0067.png not found")
inv = depth_net.infer_image(img)
rel = inv.max() - inv
scale = 0.2 / rel[int(cy), int(cx)]
Zmap = rel * scale


def is_below_brick_surface(pt, surface_point, surface_normal):
    # Return True if the point is below the brick surface
    v = pt - surface_point
    return np.dot(v, surface_normal) < -1e-4  # small epsilon for numerical tolerance



# -------------------- Compute Brick Normal --------------------
P_cam = []
for u,v in brick_px:
    Z = Zmap[int(v), int(u)]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    P_cam.append([X,Y,Z])
P_cam = np.array(P_cam)  # (3,3)

a, b = P_cam[1]-P_cam[0], P_cam[2]-P_cam[0]
brick_n = np.cross(a, b)
brick_n /= np.linalg.norm(brick_n)
brick_n = -brick_n
print("brick normal =", brick_n)

# -------------------- Load Trowel Tip Vertices --------------------
pts = torch.load("trowel_tip_vertex_trajectories_3d.pt")  # (3,T,3)
pts = pts.numpy() if isinstance(pts, torch.Tensor) else pts
N, T, _ = pts.shape

a = pts[1] - pts[0]
b = pts[2] - pts[0]
tri_n = np.cross(a, b)
tri_n = tri_n / np.linalg.norm(tri_n, axis=1, keepdims=True)
centers = pts.mean(0)  # (T,3)

# -------------------- Brick Geometry --------------------
brick_size = np.array([0.095, 0.045, 0.02])  # L,W,H in meters

def get_brick_mesh(center_top, normal, size):
    """Generate 8 vertices of brick centered at center_top (on top surface)"""
    z = normal / np.linalg.norm(normal)
    # define arbitrary x-axis on brick top
    ref = np.array([1,0,0]) if abs(z @ [1,0,0]) < 0.9 else np.array([0,1,0])
    x = np.cross(ref, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1)  # (3,3)
    brick_center = center_top - z * (size[2] / 2)
    
    # 8 vertices in local frame
    dx, dy, dz = size / 2
    corners_local = np.array([
        [-dx,-dy,-dz], [ dx,-dy,-dz], [ dx, dy,-dz], [-dx, dy,-dz],
        [-dx,-dy, dz], [ dx,-dy, dz], [ dx, dy, dz], [-dx, dy, dz],
    ])
    corners_world = brick_center[None,:] + corners_local @ R.T
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
    return corners_world, faces

# -------------------- Visualization --------------------
fig = plt.figure(figsize=(6,5))
ax  = fig.add_subplot(111, projection='3d')

def draw(k):
    ax.cla()
    sc = np.linalg.norm(pts[0,k]-pts[1,k])*0.6

    # --- brick top center (must be defined first) ---
    brick_top_center = pts[1, k]  # Keypoint 1 always on brick top

    # --- Triangle ---
    tri = Poly3DCollection([pts[:,k]], alpha=0.3, facecolor='limegreen')
    ax.add_collection3d(tri)

    # --- Triangle Vertex IDs ---
    for i in range(3):
        x, y, z = pts[i, k]
        label = f'{i}'
        if is_below_brick_surface(pts[i, k], brick_top_center, brick_n):
            label += ' (penetrate)'
            ax.scatter(x, y, z, color='red', s=50)
        ax.text(x, y, z + 0.005, label, color='black', fontsize=10)

    # --- Normals ---
    ax.quiver(*centers[k], *(tri_n[k]*sc), color='red', linewidth=2)
    ax.quiver(*centers[k], *(brick_n*sc), color='blue', linewidth=2)

    # --- Brick ---
    verts, faces = get_brick_mesh(brick_top_center, brick_n, brick_size)
    brick = Poly3DCollection([verts[f] for f in faces], alpha=0.3, facecolor='orange')
    ax.add_collection3d(brick)

    # --- Camera axes ---
    axis_len = sc*1.5
    ax.quiver(0,0,0, axis_len,0,0, color='r')
    ax.quiver(0,0,0, 0,axis_len,0, color='g')
    ax.quiver(0,0,0, 0,0,axis_len, color='b')
    ax.text(axis_len,0,0,'X',color='r')
    ax.text(0,axis_len,0,'Y',color='g')
    ax.text(0,0,axis_len,'Z',color='b')

    # --- Config ---
    ax.set_title(f"Frame {k}/{T-1}")
    allp = pts.reshape(-1,3); pad=(allp.max(0)-allp.min(0))*0.1
    ax.set_xlim(allp[:,0].min()-pad[0], allp[:,0].max()+pad[0])
    ax.set_ylim(allp[:,1].min()-pad[1], allp[:,1].max()+pad[1])
    ax.set_zlim(allp[:,2].min()-pad[2], allp[:,2].max()+pad[2])
    ax.set_xlabel('X (right)'); ax.set_ylabel('Y (down)'); ax.set_zlabel('Z (forward)')
    ax.set_box_aspect([1,1,1])
    ax.text2D(0.02,0.95,"red = trowel n   blue = brick n", transform=ax.transAxes)


idx=[0]
def on_key(e):
    if e.key in ('right','left'):
        idx[0] = (idx[0]+1)%T if e.key=='right' else (idx[0]-1)%T
        draw(idx[0])
    elif e.key=='escape':
        plt.close(fig)

fig.canvas.mpl_connect('key_press_event', on_key)
draw(0)
plt.show()
