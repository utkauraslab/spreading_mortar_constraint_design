
"""
Extract each frame's trowel's polygon area vertices from the json file.
Each json file is saved by labelme which created polygon region on the trowel for each frame, representing the 
canonical shape of the trowel.

'trowel_polygon_vertices.py' contains the polygon region vertices 2D coordinates frame by frame

"""


import os
import json
import numpy as np
import sys
from natsort import natsorted

# --- Configuration ---
# This script assumes it's run from your project's root directory.
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# The folder where you have saved all your LabelMe .json files.
# Please create this folder and place your JSON files inside it.
ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "data", "labels")
ANNOTATIONS_DIR_2 = os.path.join(PROJECT_ROOT, "data", "triangle_labels_1")
# The name of the output file that will store all vertex coordinates.
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "trowel_polygon_vertices.npy")
OUTPUT_FILE_PATH_2 = os.path.join(PROJECT_ROOT, "trowel_tip_polygon_vertices.npy")


# --- Main Execution ---
if __name__ == "__main__":
    

    # Find all JSON files in the directory
    json_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.json')]
    json_files_2 = [f for f in os.listdir(ANNOTATIONS_DIR_2) if f.endswith('.json')]
    
    # Use natsort to ensure correct numerical order (e.g., frame_1, frame_2, ..                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ., frame_10)
    # This is more reliable than standard sorting for filenames with numbers.
    sorted_json_files = natsorted(json_files)
    sorted_json_files_2 = natsorted(json_files_2)

    
    # This list will hold the vertex array for each frame
    all_frames_vertices = []
    all_frames_vertices_2 = []
    # Process each file in sorted order
    for filename in sorted_json_files:
        file_path = os.path.join(ANNOTATIONS_DIR, filename)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Assuming there is at least one shape and we want the first one
            if data['shapes']:
                # Extract the list of [x, y] points
                points = data['shapes'][0]['points']
                
                # Convert the list of lists to a NumPy array of integers
                vertices_array = np.array(points, dtype=np.int32)
                
                # Add the vertices for this frame to our master list
                all_frames_vertices.append(vertices_array)
            else:
                print(f"Warning: No shapes found in {filename}. Skipping.")
                # Add an empty array to maintain frame count if needed
                all_frames_vertices.append(np.array([], dtype=np.int32))

        except Exception as e:
            print(f"Error processing file {filename}: {e}")


    for filename in sorted_json_files_2:
        file_path_2 = os.path.join(ANNOTATIONS_DIR_2, filename)
        
        try:
            with open(file_path_2, 'r') as f:
                data = json.load(f)
            
            # Assuming there is at least one shape and we want the first one
            if data['shapes']:
                # Extract the list of [x, y] points
                points = data['shapes'][0]['points']
                
                # Convert the list of lists to a NumPy array of integers
                vertices_array = np.array(points, dtype=np.int32)
                
                # Add the vertices for this frame to our master list
                all_frames_vertices_2.append(vertices_array)
            else:
                print(f"Warning: No shapes found in {filename}. Skipping.")
                # Add an empty array to maintain frame count if needed
                all_frames_vertices_2.append(np.array([], dtype=np.int32))

        except Exception as e:
            print(f"Error processing file {filename}: {e}")


    # # Save the list of NumPy arrays to a single .npy file
    # # The resulting .npy file will have dtype=object because the arrays may have different lengths
    # np.save(OUTPUT_FILE_PATH, np.array(all_frames_vertices, dtype=object))
    # print(np.array(all_frames_vertices, dtype=object).shape)
    # np.save(OUTPUT_FILE_PATH_2, np.array(all_frames_vertices_2, dtype=object))
    # print(np.array(all_frames_vertices_2, dtype=object).shape)

    # print(f"\nProcessing complete.")
    # print(f"Successfully saved vertex data for {len(all_frames_vertices)} frames to:")
    # print(OUTPUT_FILE_PATH)
    # t1 = np.array(all_frames_vertices, dtype=object)
    # print(t1[0])
    # t2 = np.array(all_frames_vertices_2, dtype=object)
    # print(t2[0])


    def as_object_array(list_of_polys):
        """Return a 1-D object array where each element is an (N,2) int32 array (or empty (0,2))."""
        obj = np.empty(len(list_of_polys), dtype=object)
        for i, poly in enumerate(list_of_polys):
            arr = np.asarray(poly)
            if arr.ndim != 2 or arr.shape[1] != 2:
                # sanitize to empty (0,2) if anything odd
                arr = np.zeros((0, 2), dtype=np.int32)
            else:
                arr = arr.astype(np.int32, copy=False)
            obj[i] = arr
        return obj

    # Force both to aligned 1-D object arrays
    arr_body = as_object_array(all_frames_vertices)
    arr_tip  = as_object_array(all_frames_vertices_2)

    # Optional sanity checks:
    # - same length
    assert arr_body.shape == arr_tip.shape == (len(all_frames_vertices),)

    # Save exactly as 1-D object arrays
    np.save(OUTPUT_FILE_PATH, arr_body, allow_pickle=True)
    np.save(OUTPUT_FILE_PATH_2, arr_tip, allow_pickle=True)

    # Debug prints
    print(arr_body.shape)   # (70,)
    print(arr_tip.shape)    # (70,)
    print("\nProcessing complete.")
    print(f"Successfully saved vertex data for {len(arr_body)} frames to:")
    print(OUTPUT_FILE_PATH)
    print(arr_body[0])      # first polygon (N,2)
    print(arr_tip[0])       # first tip polygon (M,2)
    
