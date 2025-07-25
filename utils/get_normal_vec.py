# visualize_tip_normals.py
import torch, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl


pts = torch.load("trowel_tip_vertex_trajectories_3d.pt")  # (3,T,3)
if isinstance(pts, torch.Tensor):
    pts = pts.numpy()
elif isinstance(pts, np.ndarray):
    pass
else:
    raise TypeError("unexpected type")

N_pts, T, _ = pts.shape



normals = []
centers = []
for k in range(T):
    p1, p2, p3 = pts[:, k]          # (3,3)
    a, b = p2 - p1, p3 - p1
    n = np.cross(a, b)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        n_unit = np.array([np.nan]*3)   
    else:
        n_unit = n / norm
    normals.append(n_unit)
    centers.append((p1 + p2 + p3) / 3)

normals  = np.stack(normals)   # (T,3)
centers  = np.stack(centers)   # (T,3)
torch.save(torch.tensor(normals), "trowel_tip_vertex_region_normals.pt")
print("saved normals  ->  trowel_tip_vertex_region_normals.pt  shape:", normals.shape)


mpl.rcParams["toolbar"] = "None"
fig = plt.figure(figsize=(5,5))
ax  = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_box_aspect([1,1,1])

tri_artist = None
arrow      = None
frame_idx  = [0]      # use list for mutability

# def draw_frame(k):
#     global tri_artist, arrow
#     ax.cla()
#     p = pts[:, k]
#     center = centers[k]
#     nvec   = normals[k]

    
#     tri_artist = Poly3DCollection([p], alpha=0.3, facecolor="limegreen")
#     ax.add_collection3d(tri_artist)

    
#     scale = np.linalg.norm(p[0]-p[1]) * 0.6
#     arrow = ax.quiver(*center, *nvec*scale, color="red")

#     ax.set_title(f"Frame {k:02d}")
    
#     whole = pts.reshape(-1,3)
#     for dim in range(3):
#         mn, mx = whole[:,dim].min(), whole[:,dim].max()
#         pad = (mx-mn)*0.1
#         ax.set_xlim(whole[:,0].min()-pad, whole[:,0].max()+pad)
#         ax.set_ylim(whole[:,1].min()-pad, whole[:,1].max()+pad)
#         ax.set_zlim(whole[:,2].min()-pad, whole[:,2].max()+pad)
#     plt.draw()



def draw_frame(k):
    global tri_artist, arrow
    ax.cla()

    
    p = pts[:, k]
    tri_artist = Poly3DCollection([p], alpha=0.3, facecolor="limegreen")
    ax.add_collection3d(tri_artist)

    
    center = centers[k]
    nvec   = normals[k]
    scale  = np.linalg.norm(p[0] - p[1]) * 0.6
    arrow  = ax.quiver(*center, *nvec*scale,
                       color="red", linewidth=2)

    
    axis_len = scale * 1.5     
    ax.quiver(0, 0, 0,  axis_len, 0, 0, color='r', linewidth=2)
    ax.quiver(0, 0, 0,  0, axis_len, 0, color='g', linewidth=2)
    ax.quiver(0, 0, 0,  0, 0, axis_len, color='b', linewidth=2)
    ax.text(axis_len,   0,   0, 'X', color='r')
    ax.text(0, axis_len,   0, 'Y', color='g')
    ax.text(0,   0, axis_len, 'Z', color='b')

    
    ax.set_title(f"Frame {k:02d}")
    whole = pts.reshape(-1,3)
    pad   = (whole.max(0)-whole.min(0))*0.1
    ax.set_xlim(whole[:,0].min()-pad[0], whole[:,0].max()+pad[0])
    ax.set_ylim(whole[:,1].min()-pad[1], whole[:,1].max()+pad[1])
    ax.set_zlim(whole[:,2].min()-pad[2], whole[:,2].max()+pad[2])
    ax.set_xlabel('X (right)')
    ax.set_ylabel('Y (down)')
    ax.set_zlabel('Z (forward)')
    ax.set_box_aspect([1,1,1])
    plt.draw()


def on_key(event):
    if event.key in ("right", "left"):
        if event.key == "right":
            frame_idx[0] = (frame_idx[0] + 1) % T
        else:
            frame_idx[0] = (frame_idx[0] - 1) % T
        draw_frame(frame_idx[0])
    elif event.key == "escape":
        plt.close(fig)

fig.canvas.mpl_connect("key_press_event", on_key)
draw_frame(0)

plt.show()
