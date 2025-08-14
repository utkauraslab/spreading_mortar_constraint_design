import os
import sys
import numpy as np
from tqdm import tqdm


"""
Use static ground pixel areas to refine depth maps for temporal consistency.

'depth_map_cross_frames_refined.npy' stores the refined depth map cross frames which is temporal consistent.


"""


# paths 
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames.npy")
GROUND_REGION_PATH = os.path.join(PROJECT_ROOT, "seg_data", "ground_region.npy")


# output file path
REFINED_DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")



if __name__ == "__main__":
    # Load
    original_depth_maps = np.load(DEPTH_MAP_PATH)
    ground_coords = np.load(GROUND_REGION_PATH)
    

    num_frames, height, width = original_depth_maps.shape

    # Mask for the static-ground region 
    ground_mask = np.zeros((height, width), dtype=bool)
    # Ensure the coordinates are within the image bounds before creating the mask
    valid_coords = (ground_coords[:, 1] < height) & (ground_coords[:, 0] < width)
    coords = ground_coords[valid_coords]
    # NumPy indexing is (row, col), which corresponds to (y, x)
    ground_mask[coords[:, 1], coords[:, 0]] = True



    # Extract depth for the static-ground region in the 1st frame
    depth_map_frame0 = original_depth_maps[0]
    ground_depths_frame0 = depth_map_frame0[ground_mask]

    
    # get mean depth value for the static-ground region
    reference_mean_depth = np.mean(ground_depths_frame0)
    print(f"Reference mean depth from frame 0's ground region: {reference_mean_depth:.4f}m")

    # Refine subsequent frames 
    refined_depth_maps = np.zeros_like(original_depth_maps)
    refined_depth_maps[0] = depth_map_frame0
    
    print(reference_mean_depth)
    print("following:")
    for i in tqdm(range(1, num_frames), desc="Refining frames"):
        current_depth_map = original_depth_maps[i]
        
        # Get the depth values for the ground region in the current frame
        ground_depths_current = current_depth_map[ground_mask]
        # get mean depth value for current static-ground region
        current_mean_depth = np.mean(ground_depths_current)
        
        #print(f"frame {i}: mean depth = {current_mean_depth}")
        
        # Calculate the scale factor needed to match the reference depth
        # Avoid division by zero
        if current_mean_depth > 1e-6:
            scale_factor = current_mean_depth / reference_mean_depth
        else:
            scale_factor = 1.0 # No change if current depth is zero
            
        #Apply the scale factor to the *entire* depth map for the current frame
        refined_depth_maps[i] = current_depth_map * (1.0 / scale_factor)
        #refined_depth_maps[i] = current_depth_map * scale_factor

    print("Refinement complete.")

   

    # save the refined depth maps
    np.save(REFINED_DEPTH_MAP_PATH, refined_depth_maps)
    print(f"\nSaved refined depth maps to: {REFINED_DEPTH_MAP_PATH}")
    


