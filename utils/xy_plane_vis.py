# # visualize_tip_xy_triangle.py
# import torch, numpy as np, matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation


# verts = torch.load("trowel_tip_vertex_trajectories_3d.pt")   # (3, T, 3)
# if isinstance(verts, torch.Tensor):
#     verts = verts.numpy()
# N, T, _ = verts.shape         # N=3


# xy = verts[:, :, :2]          # (3, T, 2)


# fig, ax = plt.subplots(figsize=(5, 5))
# tri_line, = ax.plot([], [], 'o-', lw=2, color='limegreen')
# title = ax.text(0.02, 0.95, '', transform=ax.transAxes)


# all_xy = xy.reshape(-1, 2)
# pad = (all_xy.max(0) - all_xy.min(0)) * 0.1
# ax.set_xlim(all_xy[:, 0].min() - pad[0], all_xy[:, 0].max() + pad[0])
# ax.set_ylim(all_xy[:, 1].min() - pad[1], all_xy[:, 1].max() + pad[1])

# ax.invert_yaxis()
# ax.set_aspect('equal')
# ax.set_xlabel('X (right)'); ax.set_ylabel('Y (down)')
# ax.set_title('Trowel tip triangle in camera XY plane')


# def init():
#     tri_line.set_data([], [])
#     title.set_text('')
#     return tri_line, title

# def update(frame):
#     pts = xy[:, frame]                    # (3,2)
    
#     closed = np.vstack([pts, pts[0]])
#     tri_line.set_data(closed[:, 0], closed[:, 1])
#     title.set_text(f'Frame {frame}/{T-1}')
#     return tri_line, title

# ani = FuncAnimation(fig, update, frames=T, init_func=init,
#                     interval=120, blit=True, repeat=True)

# plt.show()







# visualize_tip_xy_triangle.py
import torch, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

verts = torch.load("trowel_tip_vertex_trajectories_3d.pt")   # (3, T, 3)
if isinstance(verts, torch.Tensor):
    verts = verts.numpy()
N, T, _ = verts.shape         # N=3

xy = verts[:, :, :2]          # (3, T, 2)

fig, ax = plt.subplots(figsize=(5, 5))
tri_line, = ax.plot([], [], 'o-', lw=2, color='limegreen')
title = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Set axis limits
flat = xy.reshape(-1, 2)
pad = np.ptp(flat, axis=0) * 0.1
ax.set_xlim(flat[:, 0].min() - pad[0], flat[:, 0].max() + pad[0])
ax.set_ylim(flat[:, 1].min() - pad[1], flat[:, 1].max() + pad[1])

ax.invert_yaxis()
ax.set_aspect('equal')
ax.set_xlabel('X (right)')
ax.set_ylabel('Y (down)')
ax.set_title('Trowel tip triangle in camera XY plane')

# Create 3 text label placeholders for vertex ids
labels = [ax.text(0, 0, '', fontsize=9, color='blue') for _ in range(3)]

def init():
    tri_line.set_data([], [])
    title.set_text('')
    for label in labels:
        label.set_text('')
        label.set_position((0, 0))
    return [tri_line, title] + labels

def update(frame):
    pts = xy[:, frame]                    # (3,2)
    closed = np.vstack([pts, pts[0]])
    tri_line.set_data(closed[:, 0], closed[:, 1])
    title.set_text(f'Frame {frame}/{T-1}')

    for i in range(3):
        labels[i].set_text(str(i))
        labels[i].set_position((pts[i, 0], pts[i, 1]))

    return [tri_line, title] + labels

ani = FuncAnimation(fig, update, frames=T, init_func=init,
                    interval=120, blit=True, repeat=True)

plt.show()
