import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D         # noqa: F401
from matplotlib.animation import FuncAnimation


R = np.array([[ 0.6905125,   0.4495278,  -0.5666722 ],
 [ 0.67629963, -0.6790981,   0.28538513],
 [-0.25653744, -0.5803021,  -0.7729411 ]], dtype=float)
t = np.array([0.3159029,  -0.03663168,  0.15089016], dtype=float)

T = np.eye(4)
T[:3,:3] = R
T[:3, 3] = t


traj_cam = torch.load("trowel_tip_keypoints_trajectories_3d.pt")


keypts, frames, _ = traj_cam.shape             # keypts=3, frames=70
traj_cam_h = np.concatenate(
    [traj_cam.reshape(-1, 3), np.ones((keypts*frames, 1))], axis=1)  # (3F,4)


traj_brick_h = (T @ traj_cam_h.T).T            # (3F,4)
traj_brick   = traj_brick_h[:, :3].reshape(keypts, frames, 3)


fig = plt.figure(figsize=(6, 6))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("Trowel tip trajectories in brick-local frame")
ax.set_xlabel("X (brick)")
ax.set_ylabel("Y (brick)")
ax.set_zlabel("Z (brick)")


axis_len = 0.05   
p0 = np.zeros(3)  # local origin
ax.quiver(*p0, *(R[0]*axis_len), color='r', linewidth=2)   # X 轴
ax.quiver(*p0, *(R[1]*axis_len), color='g', linewidth=2)   # Y 轴
ax.quiver(*p0, *(R[2]*axis_len), color='b', linewidth=2)   # Z 轴
ax.text(*(R[0]*axis_len), "X", color='r')
ax.text(*(R[1]*axis_len), "Y", color='g')
ax.text(*(R[2]*axis_len), "Z", color='b')


colors = ["tab:purple", "tab:orange", "tab:cyan"]
lines  = [ax.plot([], [], [], color=c, lw=1.5)[0] for c in colors]
points = [ax.scatter([], [], [], color=c, s=25)    for c in colors]


all_xyz = traj_brick.reshape(-1, 3)
x_min, x_max = all_xyz[:,0].min(), all_xyz[:,0].max()
y_min, y_max = all_xyz[:,1].min(), all_xyz[:,1].max()
z_min, z_max = all_xyz[:,2].min(), all_xyz[:,2].max()
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_zlim(z_min, z_max)

def update(frame_idx):
    for k in range(keypts):
        xyz_hist = traj_brick[k, :frame_idx+1]          
        lines[k].set_data(xyz_hist[:,0], xyz_hist[:,1])
        lines[k].set_3d_properties(xyz_hist[:,2])
        points[k]._offsets3d = ( [traj_brick[k,frame_idx,0]],
                                 [traj_brick[k,frame_idx,1]],
                                 [traj_brick[k,frame_idx,2]] )
    return lines + points

ani = FuncAnimation(fig, update, frames=frames,
                    interval=100, blit=True)

plt.show()

