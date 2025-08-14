import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# --- Configuration ---
# File paths (assuming the script is run from the project's root directory)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SEGMENTED_COORDS_PATH = os.path.join(PROJECT_ROOT, "seg_data", "ground_region.npy")
# DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "seg_data","depth_map_cross_frames.npy")
DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
OUTPUT_MOVIE_PATH = os.path.join(PROJECT_ROOT, "ground_region_depth_animation_refined.mp4")

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading input data...")
    # Check for all necessary files
    required_files = [SEGMENTED_COORDS_PATH, DEPTH_MAP_PATH]
    if not all(os.path.exists(p) for p in required_files):
        print("Error: One or more required data files (.npy) not found.")
        print(f"Looking for: {SEGMENTED_COORDS_PATH}")
        print(f"Looking for: {DEPTH_MAP_PATH}")
        sys.exit(1)

    # Load the data
    print("Loading .npy files...")
    segmented_coords = np.load(SEGMENTED_COORDS_PATH)
    all_depth_maps = np.load(DEPTH_MAP_PATH)
    print("Data loaded successfully.")

    num_frames, height, width = all_depth_maps.shape
    print(f"Found {num_frames} frames with resolution {width}x{height}.")

    # --- 1. Create a Mask for the Segmented Region ---
    # The mask will be True only at the coordinates of our ground region.
    region_mask = np.zeros((height, width), dtype=bool)
    # NumPy indexing is (row, col), which corresponds to (y, x)
    # We need to ensure the coordinates are within the image bounds
    valid_coords = (segmented_coords[:, 1] < height) & (segmented_coords[:, 0] < width)
    coords = segmented_coords[valid_coords]
    region_mask[coords[:, 1], coords[:, 0]] = True

    # --- 2. Determine Global Depth Range for Consistent Coloring ---
    # Extract all depth values within the masked region across all frames
    depths_in_region = all_depth_maps[:, region_mask]
    
    # Find the min and max depth values to create a consistent color scale
    global_min_depth = np.nanmin(depths_in_region)
    global_max_depth = np.nanmax(depths_in_region)
    print(f"Global depth range for region: {global_min_depth:.4f}m to {global_max_depth:.4f}m")

    # --- 3. Set up the Animation Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('dark_background')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')

    # Create a colormap and normalization object
    colormap = plt.get_cmap('viridis')
    norm = Normalize(vmin=global_min_depth, vmax=global_max_depth)

    # Create an initial RGBA image. We use 4 channels for Red, Green, Blue, Alpha (transparency).
    # Start with a fully transparent image.
    rgba_image = np.zeros((height, width, 4))
    im = ax.imshow(rgba_image)

    # Add a color bar to the plot
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=colormap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # --- 4. Define the Animation Update Function ---
    def update(frame_index):
        # Get the depth map for the current frame
        depth_frame = all_depth_maps[frame_index]
        
        # Create an array filled with NaN to hold only the region's depths
        region_depth_values = np.full((height, width), np.nan)
        # Populate it with the actual depth values from the mask
        region_depth_values[region_mask] = depth_frame[region_mask]
        
        # Normalize the depth values to the range [0, 1] for coloring
        normalized_depths = norm(region_depth_values)
        
        # Apply the colormap to get RGBA values
        rgba_output = colormap(normalized_depths)
        
        # Make the background (where depth is NaN) fully transparent
        rgba_output[~region_mask] = [0, 0, 0, 0]
        
        # Update the image data
        im.set_data(rgba_output)
        
        # Update the title
        ax.set_title(f'Depth of Ground Region - Frame {frame_index + 1}/{num_frames}', color='white')
        return [im]

    # --- 5. Create and Save the Animation ---
    print("Creating and saving animation... This may take a moment.")
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
    
    # Save the animation as an MP4 file
    ani.save(OUTPUT_MOVIE_PATH, writer='ffmpeg', fps=10, dpi=150)
    
    print(f"Successfully saved animation to {OUTPUT_MOVIE_PATH}")
    # plt.show() # Uncomment to display the animation interactively
