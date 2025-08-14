
"""
Use estimated camera intrinsic to project each frame's trowel polygon region's vertices into 3D coordinates in camera frame and store.

'trowel_polygon_vertices_3d.py' contains each frame's trowel polygon region's vertices 3D coordinates in cam frame.

"""



import os
import sys
import numpy as np

# --- Configuration ---
# Camera intrinsic parameters
FX = 836.0
FY = 836.0
CX = 979.0
CY = 632.0
INTRINSICS = np.array([FX, FY, CX, CY])

# File paths (assuming the script is run from the project's root directory)
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
#KEYPOINTS_2D_PATH = os.path.join(PROJECT_ROOT, "seg_data","keypoints_2d_traj.npy") 
REFINED_DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
TROWEL_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
TROWEL_TIP_VERTICES_2D_PATH = os.path.join(PROJECT_ROOT, "trowel_tip_polygon_vertices.npy")

# Output Paths
#OUTPUT_3D_TRAJ_PATH = os.path.join(PROJECT_ROOT, "keypoints_3d_traj_refined.npy")
OUTPUT_3D_VERTICES_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices_3d.npy")
OUTPUT_3D_TIP_VERTICES_PATH = os.path.join(PROJECT_ROOT, "trowel_tip_polygon_vertices_3d.npy")

# --- Helper Functions ---

def unproject_points(coords_2d_list, refined_depth_maps, intrinsics):
    """
    Projects a list of 2D vertex arrays (one for each frame) into a list of 3D vertex arrays.
    """
    fx, fy, cx, cy = intrinsics
    num_frames = len(refined_depth_maps)
    all_frames_vertices_3d = []

    for f in range(num_frames):
        frame_vertices_2d = coords_2d_list[f]
        frame_vertices_3d = []
        
        if frame_vertices_2d.size > 0:
            for u, v in frame_vertices_2d:
                u, v = int(u), int(v)
                height, width = refined_depth_maps[f].shape
                if 0 <= v < height and 0 <= u < width:
                    Z = refined_depth_maps[f, v, u]
                    if Z > 0 and not np.isnan(Z):
                        X = (u - cx) * Z / fx
                        Y = (v - cy) * Z / fy
                        frame_vertices_3d.append([X, Y, Z])
                    else:
                        frame_vertices_3d.append([np.nan, np.nan, np.nan])
                else:
                    frame_vertices_3d.append([np.nan, np.nan, np.nan])
        
        all_frames_vertices_3d.append(np.array(frame_vertices_3d))
        
    return np.array(all_frames_vertices_3d, dtype=object)


# --- Main Execution ---
if __name__ == "__main__":
    print("Loading input data...")
    # Load the data
    refined_depth_maps = np.load(REFINED_DEPTH_MAP_PATH)
    #keypoints_2d = np.load(KEYPOINTS_2D_PATH)
    trowel_vertices_2d = np.load(TROWEL_VERTICES_2D_PATH, allow_pickle=True)
    trowel_tip_vertices_2d = np.load(TROWEL_TIP_VERTICES_2D_PATH, allow_pickle=True)

    num_frames, height, width = refined_depth_maps.shape
    #num_keypoints, _, _ = keypoints_2d.shape

    # --- Task 1: Project 2D keypoints to 3D using refined depth maps ---
    # print("\nTask 1: Projecting 2D keypoints to 3D...")
    # trajectories_3d_refined = np.zeros((num_keypoints, num_frames, 3))

    # for k in range(num_keypoints):
    #     for f in range(num_frames):
    #         u, v = keypoints_2d[k, f]
    #         u, v = int(u), int(v)
            
    #         if 0 <= v < height and 0 <= u < width:
    #             Z = refined_depth_maps[f, v, u]
    #             if Z > 0 and not np.isnan(Z):
    #                 X = (u - CX) * Z / FX
    #                 Y = (v - CY) * Z / FY
    #                 trajectories_3d_refined[k, f] = [X, Y, Z]
    #             else:
    #                 trajectories_3d_refined[k, f] = [np.nan, np.nan, np.nan]
    #         else:
    #             trajectories_3d_refined[k, f] = [np.nan, np.nan, np.nan]

    # np.save(OUTPUT_3D_TRAJ_PATH, trajectories_3d_refined)
    # print(f"Successfully saved refined 3D keypoint trajectories.")
    # print(f"--> Output file: {os.path.basename(OUTPUT_3D_TRAJ_PATH)}")
    # print(f"    Output shape: {trajectories_3d_refined.shape}")

    # --- Task 2: Project 2D Trowel Polygon Vertices to 3D ---
    print("\nTask 2: Projecting 2D trowel polygon vertices to 3D...")
    all_frames_vertices_3d = unproject_points(trowel_vertices_2d, refined_depth_maps, INTRINSICS)
    np.save(OUTPUT_3D_VERTICES_PATH, all_frames_vertices_3d)
    print(f"Successfully saved refined 3D polygon vertices.")
    print(f"--> Output file: {os.path.basename(OUTPUT_3D_VERTICES_PATH)}")
    print(f"    Output contains {len(all_frames_vertices_3d)} frames.")

    all_frames_tip_vertices_3d = unproject_points(trowel_tip_vertices_2d, refined_depth_maps, INTRINSICS)
    np.save(OUTPUT_3D_TIP_VERTICES_PATH, all_frames_tip_vertices_3d)
    print(f"Successfully saved refined 3D polygon vertices.")
    print(f"--> Output file: {os.path.basename(OUTPUT_3D_TIP_VERTICES_PATH)}")
    print(f"    Output contains {len(all_frames_tip_vertices_3d)} frames.")

    # --- Task 3: Project 2D Trowel Tip Vertices to 3D ---
    #print("\nTask 3: Projecting 2D trowel tip vertices to 3D...")
    #all_frames_tip_vertices_3d = unproject_points(trowel_tip_vertices_2d, refined_depth_maps, INTRINSICS)
    #np.save(OUTPUT_3D_TIP_VERTICES_PATH, all_frames_tip_vertices_3d)
    # print(f"Successfully saved refined 3D trowel tip vertices.")
    # print(f"--> Output file: {os.path.basename(OUTPUT_3D_TIP_VERTICES_PATH)}")
    # print(f"    Output contains {len(all_frames_tip_vertices_3d)} frames.")
