



# import os
# import cv2
# import torch
# import numpy as np

# # -------------------- Load Data --------------------
# # Brick 3D trajectory (3, T, 3)
# pts = torch.load("trowel_tip_vertex_trajectories_3d.pt")  # keypoints 3D
# pts = pts.numpy() if isinstance(pts, torch.Tensor) else pts
# N, T, _ = pts.shape

# # Keypoint 2D and depth (3, T) and (3, T, 2)
# keypoints_2d = torch.load("trowel_tip_vertex_pred_tracks.pt").numpy()
# depth_values = torch.load("trowel_tip_vertex_depth_tensor.pt").numpy()

# # -------------------- Camera Intrinsics --------------------
# fx = fy = 836.0
# cx, cy = 979.0, 632.0
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
# img_h, img_w = 1264, 1958
# rvec = np.zeros((3, 1), dtype=np.float32)
# tvec = np.zeros((3, 1), dtype=np.float32)

# # -------------------- Brick Geometry --------------------
# brick_size = np.array([0.095, 0.045, 0.02])  # L, W, H (meters)

# def get_brick_mesh(center_top, normal, size):
#     z = normal / np.linalg.norm(normal)
#     ref = np.array([1, 0, 0]) if abs(z @ [1, 0, 0]) < 0.9 else np.array([0, 1, 0])
#     x = np.cross(ref, z); x /= np.linalg.norm(x)
#     y = np.cross(z, x)
#     R = np.stack([x, y, z], axis=1)
#     brick_center = center_top - z * (size[2] / 2)
#     dx, dy, dz = size / 2
#     corners_local = np.array([
#         [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
#         [-dx, -dy, dz],  [dx, -dy, dz],  [dx, dy, dz],  [-dx, dy, dz],
#     ])
#     return brick_center[None, :] + corners_local @ R.T  # (8, 3)

# # -------------------- Output Folder --------------------
# image_folder = "./bricklaying_data"
# output_dir = "./fused_backproj_annotated"
# os.makedirs(output_dir, exist_ok=True)

# # -------------------- Process All Frames --------------------
# for f in range(T):
#     # 1. Compute brick 8 corners from 3D keypoints
#     a = pts[1, f] - pts[0, f]
#     b = pts[2, f] - pts[0, f]
#     brick_n = -np.cross(a, b)
#     brick_n /= np.linalg.norm(brick_n)
#     brick_top_center = pts[1, f]
#     brick_verts3d = get_brick_mesh(brick_top_center, brick_n, brick_size)
#     brick_verts2d, _ = cv2.projectPoints(brick_verts3d.astype(np.float32), rvec, tvec, K, None)
#     brick_verts2d = brick_verts2d.reshape(-1, 2)

#     # 2. Project 3D keypoints back to 2D
#     reprojected_keypoints_2d = []
#     for k in range(N):
#         u, v = keypoints_2d[k, f]
#         z = depth_values[k, f]
#         x = (u - cx) * z / fx
#         y = (v - cy) * z / fy
#         point3d = np.array([[x, y, z]], dtype=np.float32).reshape(1, 1, 3)
#         projected, _ = cv2.projectPoints(point3d, rvec, tvec, K, None)
#         reprojected_keypoints_2d.append(projected[0, 0])
#     reprojected_keypoints_2d = np.stack(reprojected_keypoints_2d, axis=0)

#     # 3. Load image and annotate
#     frame_path = os.path.join(image_folder, f"frame_{f:04d}.png")
#     img = cv2.imread(frame_path)
#     if img is None:
#         print(f"Failed to load {frame_path}")
#         continue

#     # Draw brick corners (green)
#     for i, (u, v) in enumerate(brick_verts2d):
#         u, v = int(round(u)), int(round(v))
#         if 0 <= u < img_w and 0 <= v < img_h:
#             cv2.circle(img, (u, v), 4, (0, 255, 0), -1)
#             cv2.putText(img, f'B{i}', (u+4, v-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

#     # Draw reprojected keypoints (red)
#     for i, (u, v) in enumerate(reprojected_keypoints_2d):
#         u, v = int(round(u)), int(round(v))
#         if 0 <= u < img_w and 0 <= v < img_h:
#             cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
#             cv2.putText(img, f'K{i+1}', (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

#     # Save annotated image
#     out_path = os.path.join(output_dir, f"frame_{f:04d}_fused.png")
#     cv2.imwrite(out_path, img)
    





import numpy as np
import cv2
import torch
from depth_anything.depth_anything_v2.dpt import DepthAnythingV2


fx = fy = 836
cx, cy = 979, 632

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]], dtype=np.float32)


# 2D corner coordinates of the 190×40 side (unit: pixels) ===
# order: [top-left, bottom-left, top-right, bottom-right]
# Load model
encoder = 'vitl'
ckpt_path = f'depth_anything/checkpoints/depth_anything_v2_{encoder}.pth'
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))  # or 'cuda'
model = model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

# Load image for inference ===
img_path = "./bricklaying_data/frame_0000.png"
img = cv2.imread(img_path)
assert img is not None, f"Failed to load {img_path}"

# Infer depth map ===
depth_raw = model.infer_image(img)  # [H, W] in meters
depth_relative = np.max(depth_raw) - depth_raw

known_px, known_py = 979, 632

known_depth_m = 0.15  # known reference depth in meters
relative_at_known = depth_relative[known_py, known_px]
scale = known_depth_m / relative_at_known
depth_metric = depth_relative * scale  # final metric depth

# Extract real depth values for 4 brick points ===
brick_2d_pts = np.array([
    [587, 572],
    [587, 735],
    [1085, 906],
    [1085, 1106]
], dtype=np.int32)

depths = []
for (u, v) in brick_2d_pts:
    d = depth_metric[v, u]
    depths.append(d)
depths = np.array(depths, dtype=np.float32)


# Unproject to 3D (camera frame)
def unproject(u, v, z, K):
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z])

pts_3d = np.array([unproject(u, v, z, K) for (u, v), z in zip(brick_2d_pts, depths)])  # shape: (4, 3)

# Define brick local 3D shape (origin at top-left corner of face)
L, H, W = 0.190, 0.040, 0.090  # meters
brick_local_corners = np.array([
    [0, 0, 0],         # top-left front
    [0, H, 0],         # bottom-left front
    [L, 0, 0],         # top-right front
    [L, H, 0],         # bottom-right front
], dtype=np.float32)  # shape: (4, 3)

# Estimate rigid transform using Umeyama method
def umeyama_alignment(src, dst):
    mu_src = src.mean(0)
    mu_dst = dst.mean(0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = dst_centered.T @ src_centered / len(src)
    U, _, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = mu_dst - R @ mu_src
    return R, t

R, t = umeyama_alignment(brick_local_corners, pts_3d)


# Define full 8-corner brick shape in local frame
brick_box_local = np.array([
    [0, 0, 0],             # top-left-front
    [0, H, 0],             # bottom-left-front
    [L, 0, 0],             # top-right-front
    [L, H, 0],             # bottom-right-front
    [0, 0, W],             # top-left-back
    [0, H, W],             # bottom-left-back
    [L, 0, W],             # top-right-back
    [L, H, W],             # bottom-right-back
], dtype=np.float32)  # shape: (8, 3)

# Transform all points to camera frame
brick_box_camera = (R @ brick_box_local.T).T + t  # shape: (8, 3)



# Project back to 2D to visualize
def project_point(P_cam, K):
    x, y, z = P_cam
    u = fx * x / z + cx
    v = fy * y / z + cy
    return int(u), int(v)

# Visualize on original image
img_path = "./bricklaying_data/frame_0000.png"  # Replace with actual
img = cv2.imread(img_path)

for i, pt in enumerate(brick_box_camera):
    u, v = project_point(pt, K)
    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
        cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i+1), (u + 5, v - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Save annotated image
cv2.imwrite("annotated_brick_pose.png", img)
print("Saved: annotated_brick_pose.png")
