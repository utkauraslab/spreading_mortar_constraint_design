import cv2, torch, numpy as np
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2


fx = fy = 836.0
cx, cy = 979.0, 632.0
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float32)

img_path = "./bricklaying_data/frame_0000.png"
encoder   = "vitl"
ckpt_path = f"./depth_anything/checkpoints/depth_anything_v2_{encoder}.pth"


brick_px = np.array([
    [405, 229],     # left-top
    [254, 346],     # left-down
    [1578, 1213],   # right-down
    [1798, 1044],   # right-top
], dtype=np.int32)


device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = {'encoder':'vitl', 'features':256, 'out_channels':[256,512,1024,1024]}
net = DepthAnythingV2(**cfg).to(device).eval()
net.load_state_dict(torch.load(ckpt_path, map_location=device))

img_bgr = cv2.imread(img_path)
assert img_bgr is not None, f"failed to load {img_path}"

with torch.no_grad():
    depth_raw = net.infer_image(img_bgr)        


depth_relative = depth_raw.max() - depth_raw

known_px, known_py   = 979, 632   # assume known coordinate
known_depth_m        = 0.2       # assume known distance for this known coordinate

scale = known_depth_m / depth_relative[known_py, known_px]
depth_metric = depth_relative * scale          



P_cam = np.empty((4,3), np.float32)
for i, (u,v) in enumerate(brick_px):
    d = depth_metric[v, u]
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    P_cam[i] = [X, Y, Z]

# --------------------------------------------------
# local_z: normal vec，local_x: plane edge projection，local_y: by cross product of x and z 
# --------------------------------------------------
z_hat = np.cross(P_cam[3]-P_cam[0], P_cam[1]-P_cam[0])
z_hat /= np.linalg.norm(z_hat)
if np.dot(z_hat, P_cam.mean(0)) > 0:
    z_hat = -z_hat

edge  = P_cam[3] - P_cam[0]
x_hat = edge - np.dot(edge, z_hat)*z_hat
x_hat /= np.linalg.norm(x_hat)

y_hat = np.cross(z_hat, x_hat);  y_hat /= np.linalg.norm(y_hat)
x_hat = np.cross(y_hat, z_hat);  x_hat /= np.linalg.norm(x_hat)  



# --------------------------------------------------
# brick_local ← camera
# --------------------------------------------------
R = np.stack([x_hat, y_hat, z_hat], axis=0)
p0 = P_cam.mean(0)
t  = -R @ p0

T = np.eye(4, dtype=np.float32)
T[:3,:3] = R
T[:3, 3] = t


orth = R @ R.T
print("RRᵀ ≈\n", orth)                 # should be close to identical mat
print("det(R) =", np.linalg.det(R))    # should close to +-1
print("x·z =", np.dot(x_hat, z_hat))   # should close to 0
print("y·z =", np.dot(y_hat, z_hat))
print("x·y =", np.dot(x_hat, y_hat))


print("=== R ===\n", R)
print("\n=== t ===\n", t)
print("\n=== T (brick <- cam) ===\n", T)

