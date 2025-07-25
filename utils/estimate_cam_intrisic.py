

"""
    Estimate the camera intrinsic from the demostration video
"""



# # import cv2

# # def click_event(event, x, y, flags, param):
# #     if event == cv2.EVENT_LBUTTONDOWN:
# #         print(f"Clicked at: x={x}, y={y}")

# # # Load your image
# # img = cv2.imread('./bricklaying_data/frame_0056.png')
# # cv2.imshow('Image', img)
# # cv2.setMouseCallback('Image', click_event)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()




# # import numpy as np
# # import cv2

# # vertex_points = np.array([
# #     [0,0,0],            # top left
# #     [40,0,0],           # bottom left
# #     [0, 190, 0],        # top right
# #     [40, 190, 0],       # bottom right
# # ],dtype=np.float32)


# # image_points = [
# #     np.array([[598, 586], [598, 729], [1058, 905], [1058, 1048]], dtype=np.float32),  # Image 1: frame55
# #     np.array([[598, 586], [598, 729], [1058, 905], [1058, 1048]], dtype=np.float32),  # Image 1: frame56
# #     np.array([[598, 586], [598, 729], [1058, 905], [1058, 1048]], dtype=np.float32),  # Image 1: frame57
# #     np.array([[598, 586], [598, 729], [1058, 905], [1058, 1048]], dtype=np.float32),  # Image 1: frame58
# #     np.array([[598, 586], [598, 729], [1058, 905], [1058, 1048]], dtype=np.float32),  # Image 1: frame59
# #     # np.array([[596, 599], [597, 746]], dtype=np.float32),  # Image 2: frame56
# #     # np.array([[595, 598], [595, 745]], dtype=np.float32),  # Image 3: frame57
# #     # np.array([[597, 599], [597, 746]], dtype=np.float32),  # Image 4: frame58
# #     # np.array([[597, 601], [597, 732]], dtype=np.float32),  # Image 5: frame59
# #     # np.array([[596, 595], [596, 742]], dtype=np.float32),  # Image 6: frame60
# #     # np.array([[596, 590], [595, 727]], dtype=np.float32)   # Image 7: frame61
# #     # np.array([[596, 590], [595, 727]], dtype=np.float32)   # Image 8: frame62
# #     # np.array([[596, 590], [595, 727]], dtype=np.float32)   # Image 9: frame63
# #     # np.array([[596, 590], [595, 727]], dtype=np.float32)   # Image 10: frame64
    
# # ]


# # object_points = [vertex_points] * len(image_points)


# # image_size = (1958, 1294) 

# # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
# #     object_points, image_points, image_size, None, None
# # )


# # print("Reprojection Error:", ret)
# # print("Camera Matrix:\n", camera_matrix)
# # print("Distortion Coefficients:\n", dist_coeffs)




# import numpy as np
# from scipy.optimize import minimize

# # Sample data: Replace with your actual 2D-3D correspondences
# # 2D points: [(u1, v1), (u2, v2), ...]
# points_2d = np.array([
#     [260, 358],  # Example (u, v) coordinates
#     [261, 488],
#     [595, 607],
#     [600, 731],
#     [1060, 920],
#     [1060, 1063],
#     [1099, 949],
#     [1099, 1101]
# ])
# # 3D points: [[X1, Y1, Z1], [X2, Y2, Z2], ...]
# points_3d = np.array([
#     [-190, 0, 90],  # Example [X, Y, Z] in camera frame
#     [190, 40, 90],
#     [0, 0, 90],
#     [0, 40, 90],
#     [190, 0, 90],
#     [190, 40, 90],
#     [380, 0, 90],
#     [380, 40, 90]
# ])
# # Image dimensions
# w, h = 1958, 1294
# cx, cy = w / 2, h / 2

# def reprojection_error(params, points_3d, points_2d, cx, cy):
#     fx, fy = params
#     error = 0
#     for i in range(len(points_3d)):
#         X, Y, Z = points_3d[i]
#         u_obs, v_obs = points_2d[i]
#         u_pred = (fx * X) / Z + cx
#         v_pred = (fy * Y) / Z + cy
#         error += (u_pred - u_obs)**2 + (v_pred - v_obs)**2
#     return error

# # Initial guess for fx and fy
# initial_params = [200, 400]  # Adjust based on your image size or prior knowledge

# # Optimize
# result = minimize(
#     reprojection_error,
#     initial_params,
#     args=(points_3d, points_2d, cx, cy),
#     method='L-BFGS-B',
#     bounds=[(1, None), (1, None)]  # fx, fy > 0
# )

# fx_opt, fy_opt = result.x
# print(f"Optimized fx: {fx_opt:.2f}, fy: {fy_opt:.2f}")
# print(f"Optimization success: {result.success}, message: {result.message}")















# import cv2
# import numpy as np

# # Load the image to get its shape
# image_path = "./bricklaying_data/frame_0063.png"
# image = cv2.imread(image_path)
# if image is None:
#     raise FileNotFoundError(f"Missing image: {image_path}")
# h, w = image.shape[:2]
# print(f"Image shape: {w}x{h} pixels")

# # Provided 2D points (image coordinates in pixels)
# points_2d = np.array([
#     [260, 358],   # Example (u, v) coordinates
#     [261, 488],
#     [595, 607],
#     [600, 731],
#     [1060, 920],
#     [1060, 1063],
#     [1099, 949],
#     [1099, 1101]
# ], dtype=np.float32)

# # Provided 3D points (in mm, Z=90, custom origin)
# points_3d = np.array([
#     [-190, 0, 90],   # Origin or reference point
#     [190, 40, 90],
#     [0, 0, 90],
#     [0, 40, 90],
#     [190, 0, 90],
#     [190, 40, 90],
#     [380, 0, 90],
#     [380, 40, 90]
# ], dtype=np.float32)

# # Initial intrinsic matrix guess
# initial_K = np.array([
#     [1000.0, 0.0, w / 2],  # fx, 0, cx
#     [0.0, 1000.0, h / 2],  # 0, fy, cy
#     [0.0, 0.0, 1.0]       # 0, 0, 1
# ], dtype=np.float32)

# # Wrap into lists for calibrateCamera (single view)
# objpoints = [points_3d]
# imgpoints = [points_2d]

# # Calibrate camera with initial K
# ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), initial_K, None)

# # Output results
# print("Estimated Intrinsics (K):\n", K)
# print(f"Distortion Coefficients: {dist}")
# print(f"Rotation Vectors: {rvecs}")
# print(f"Translation Vectors: {tvecs}")
# print(f"Calibration Success (Reprojection Error): {ret}")

# # Verify reprojection error
# def compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs):
#     imgpoints2, _ = cv2.projectPoints(objpoints[0], rvecs[0], tvecs[0], K, dist)
#     error = cv2.norm(imgpoints[0], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     return error

# error = compute_reprojection_error(objpoints, imgpoints, K, dist, rvecs, tvecs)
# print(f"Mean Reprojection Error: {error:.2f} pixels")





























# import cv2
# import numpy as np

# # Image size (replace with your resolution)
# image_path = "./bricklaying_data/frame_0063.png"
# image = cv2.imread(image_path)
# if image is None:
#     raise FileNotFoundError(f"Missing image: {image_path}")
# h, w = image.shape[:2]

# # 2D image points (in pixels, from CoTracker or manual selection)


# points_2d = np.array([
#     [595, 607],
#     [600, 731],
#     [1060, 920],
#     [1060, 1063]
# ], dtype=np.float32)

# # Provided 3D points (in mm, Z=0, custom origin)
# points_3d = np.array([
#     [0, 0, 0],
#     [0, 40, 0],
#     [190, 0, 0],
#     [190, 40, 0]
# ], dtype=np.float32)


# # Initial intrinsic matrix
# initial_K = np.array([
#     [w, 0, w/2],
#     [0, w, h/2],
#     [0, 0, 1]
# ], dtype=np.float32)

# # Prepare inputs
# objpoints = [points_3d]
# imgpoints = [points_2d]

# # Calibration with constraints
# flags = (cv2.CALIB_FIX_ASPECT_RATIO |
#          cv2.CALIB_ZERO_TANGENT_DIST |
#          cv2.CALIB_FIX_PRINCIPAL_POINT)
# ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), initial_K, None, flags=flags)

# # Results
# print("Intrinsic Matrix (K):\n", K)
# print("Reprojection Error:", ret)























import cv2
import numpy as np

# Load image and get size
image_path = "./bricklaying_data/frame_0063.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Missing image: {image_path}")
h, w = image.shape[:2]
print(h)
print(w)

# # 2D image points (in pixels)
# points_2d = np.array([
#     [595, 607],
#     # [600, 731],
#     [600, 731],
#     [1060, 920],
#     [1060, 1063],
# ], dtype=np.float32)

# # 3D object points (in mm, Z=0)
# points_3d = np.array([
#     [0, 0, 0],
#     [0, 40, 0],
#     [190, 0, 0],
#     [190, 40, 0]
# ], dtype=np.float32)


points_2d = np.array([
    [260, 358],   # Example (u, v) coordinates
    [261, 488],
    [595, 607],
    [600, 731],
    [1060, 920],
    [1060, 1063],
    [1099, 949],
    [1099, 1101]
], dtype=np.float32)

# # Provided 3D points (in mm, Z=90, custom origin)
points_3d = np.array([
    [-190, 0, 0],   # Origin or reference point
    [190, 40, 0],
    [0, 0, 0],
    [0, 40, 0],
    [190, 0, 0],
    [190, 40, 0],
    [380, 0, 0],
    [380, 40, 0]
], dtype=np.float32)

# Initial intrinsic matrix
initial_K = np.array([
    [w, 0, w/2],
    [0, w, h/2],
    [0, 0, 1]
], dtype=np.float32)

# Prepare inputs
objpoints = [points_3d]
imgpoints = [points_2d]

# Calibration with additional constraints
flags = (cv2.CALIB_FIX_ASPECT_RATIO |           # f_x = f_y
         cv2.CALIB_ZERO_TANGENT_DIST |          # Tangential distortion = 0
         cv2.CALIB_FIX_PRINCIPAL_POINT |        # Fix c_x, c_y
         cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |  # Fix radial distortion
         cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 |  # Fix higher-order distortion
         cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6)   # Ensure all distortion is fixed

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), initial_K, None, flags=flags)
print("Rotation Vector (rvec):\n", rvecs[0])
print("Translation Vector (tvec):\n", tvecs[0])

# Results
print("Intrinsic Matrix (K):\n", K)
print("Reprojection Error:", ret)