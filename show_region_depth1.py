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
# --- Input Paths ---
ORIGINAL_DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "seg_data", "depth_map_cross_frames.npy")
REFINED_DEPTH_MAP_PATH = os.path.join(PROJECT_ROOT, "depth_map_cross_frames_refined.npy")
# --- Output Paths ---
ORIGINAL_MOVIE_PATH = os.path.join(PROJECT_ROOT, "full_depth_animation_original.mp4")
REFINED_MOVIE_PATH = os.path.join(PROJECT_ROOT, "full_depth_animation_refined.mp4")


def create_depth_animation(depth_maps, output_path, title_prefix):
    """
    A generic function to create and save a depth animation video.
    """
    num_frames, height, width = depth_maps.shape
    print(f"\nProcessing video: {title_prefix}")
    print(f"Found {num_frames} frames with resolution {width}x{height}.")

    # --- 1. Determine Global Depth Range for Consistent Coloring ---
    global_min_depth = np.nanmin(depth_maps)
    global_max_depth = np.nanmax(depth_maps)
    print(f"Global depth range for this video: {global_min_depth:.4f}m to {global_max_depth:.4f}m")

    # --- 2. Set up the Animation Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('dark_background')
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off')

    colormap = plt.get_cmap('viridis')
    norm = Normalize(vmin=global_min_depth, vmax=global_max_depth)

    im = ax.imshow(np.zeros((height, width)), cmap=colormap, norm=norm)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=colormap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Depth (meters)', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    # --- 3. Define the Animation Update Function ---
    def update(frame_index):
        im.set_data(depth_maps[frame_index])
        ax.set_title(f'{title_prefix} - Frame {frame_index + 1}/{num_frames}', color='white')
        return [im]

    # --- 4. Create and Save the Animation ---
    print("Creating and saving animation... This may take a moment.")
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
    
    ani.save(output_path, writer='ffmpeg', fps=10, dpi=150)
    
    print(f"Successfully saved animation to {output_path}")
    plt.close(fig) # Close the figure to free up memory


# --- Main Execution ---
if __name__ == "__main__":
    # --- Process Original (Non-Refined) Data ---
    if os.path.exists(ORIGINAL_DEPTH_MAP_PATH):
        print("Loading ORIGINAL depth maps...")
        original_depth_maps = np.load(ORIGINAL_DEPTH_MAP_PATH)
        create_depth_animation(original_depth_maps, ORIGINAL_MOVIE_PATH, "Full Scene Depth (Original)")
    else:
        print(f"Warning: Original depth map file not found at '{ORIGINAL_DEPTH_MAP_PATH}'. Skipping original video.")

    # --- Process Refined Data ---
    if os.path.exists(REFINED_DEPTH_MAP_PATH):
        print("\nLoading REFINED depth maps...")
        refined_depth_maps = np.load(REFINED_DEPTH_MAP_PATH)
        create_depth_animation(refined_depth_maps, REFINED_MOVIE_PATH, "Full Scene Depth (Refined)")
    else:
        print(f"Warning: Refined depth map file not found at '{REFINED_DEPTH_MAP_PATH}'. Skipping refined video.")
    
    print("\nAll tasks complete.")
