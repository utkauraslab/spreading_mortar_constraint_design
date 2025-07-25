
"""
    Umeyama / Kabsch to get the pose trajectory of the tool in camera frame

"""


import torch
import numpy as np
from pathlib import Path
import pickle

SRC = Path("tip_vertex_trajectories_3d.pt")
DST = Path("tool_pose_traj.pt")


try:
    raw = torch.load(SRC, map_location="cpu", weights_only=True)
except (pickle.UnpicklingError, RuntimeError):
    
    raw = torch.load(SRC, map_location="cpu")   

if isinstance(raw, np.ndarray):
    raw = torch.from_numpy(raw)
elif isinstance(raw, (list, tuple)):
    raw = torch.tensor(raw)
elif not isinstance(raw, torch.Tensor):
    raise TypeError(f"Unsupported data type: {type(raw)}")
  
raw = raw.float()        

 
if raw.shape == (3, 70, 3):
    traj_pts = raw.permute(1, 0, 2).contiguous()   # (70,3,3)
elif raw.shape == (70, 3, 3):
    traj_pts = raw.clone()
else:
    raise ValueError(f"Unexpected shape {raw.shape}, "
                     "expect (3,70,3) or (70,3,3)")

T_total, N, _ = traj_pts.shape


# freeze frame 0 as the local frame
PB = traj_pts[0]    # (3,3)

# Umeyama / Kabsch 
def rigid_transform(PB: torch.Tensor, PC: torch.Tensor):
    cB, cC = PB.mean(0), PC.mean(0)
    H = (PB - cB).T @ (PC - cC)
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:          
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = cC - R @ cB
    return R, t


eye4 = torch.eye(4)
Ts = []
for k in range(T_total):
    R, t = rigid_transform(PB, traj_pts[k])
    T = eye4.clone()
    T[:3, :3], T[:3, 3] = R, t
    Ts.append(T)

pose_traj = torch.stack(Ts)   # (70,4,4)


torch.save(pose_traj, DST)
print(f"Pose trajectory saved: {pose_traj.shape} -> {DST.resolve()}")
