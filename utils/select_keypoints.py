"""
function: Select some keypoints (to construct the sampling region)
"""

import cv2
import json
import sys # Import sys to exit gracefully

# --- Configuration ---
# Corrected path to point to the frames you extracted
# NOTE: The frame cutting script used 5-digit padding, e.g., 'frame_00000.png'
image_path = "data/frame_0000.png"
output_json = "trowel_tip_keypoints.json"


# --- Load Image with Error Handling ---
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image from path: {image_path}")
    print("Please check that the file exists and the path is correct.")
    sys.exit() # Exit the script if the image can't be loaded

clone = image.copy()
print("Image loaded successfully. Click to select points.")

# Store clicked points
clicked_points = []

# --- Mouse Callback Function ---
def click_event(event, x, y, flags, param):
    """Handles mouse click events to capture coordinates."""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        clicked_points.append({"x": x, "y": y})
        # Draw a circle on the displayed image to give visual feedback
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", clone)

# --- Main Logic ---
# Create a window and display the image
cv2.imshow("Image", clone)
# Set the mouse callback function for the "Image" window
cv2.setMouseCallback("Image", click_event)

print("Click to select point(s). Press any key to finish and save.")
# Wait until a key is pressed
cv2.waitKey(0)
# Close all OpenCV windows
cv2.destroyAllWindows()

# --- Save Points to JSON ---
with open(output_json, "w") as f:
    json.dump(clicked_points, f, indent=4)

print(f"Saved {len(clicked_points)} point(s) to {output_json}")