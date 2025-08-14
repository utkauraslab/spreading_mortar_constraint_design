
"""
    Get each brick's side surface vertices from json file and convert into 2D integer coordinate.
    Store into .npy file.

"""

import os
import json
import numpy as np
import sys

# --- Configuration ---
# This script assumes it's run from your project's root directory.
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# The path to your single LabelMe .json file that contains all the brick labels.
# Please place this file in your project's root directory or update the path.
INPUT_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "brick_side_surface_labels", "frame_0000.json")

# The directory where the individual .npy files will be saved.
# This script will create this directory if it doesn't exist.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "seg_data")






if __name__ == "__main__":
    

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Processing shapes from: {os.path.basename(INPUT_JSON_PATH)}")

    try:
        with open(INPUT_JSON_PATH, 'r') as f:
            data = json.load(f)
        
        # Check if there are any shapes in the file
        if not data['shapes']:
            print("Warning: No shapes found in the JSON file. Nothing to process.")
            sys.exit(0)

        # Iterate through each shape found in the JSON file
        for shape in data['shapes']:
            # Get the label for the current shape (e.g., "brick_side_surface_1")
            label = shape['label']
            # Get the list of [x, y] points
            points = shape['points']
            
            # Convert the list of lists to a NumPy array of integers
            vertices_array = np.array(points, dtype=np.int32)
            
            # Construct the full output path for this specific shape's .npy file
            output_filepath = os.path.join(OUTPUT_DIR, f"{label}.npy")
            
            # Save the NumPy array to its own file
            np.save(output_filepath, vertices_array)
            
            print(f" - Saved vertices for '{label}' to '{os.path.basename(output_filepath)}'")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        sys.exit(1)

    print(f"\nProcessing complete.")
    print(f"All vertex data has been saved to the '{os.path.basename(OUTPUT_DIR)}' directory.")
