
"""
    function: Select some keypoints (to construct the sampling region)

"""



import cv2
import json



# Load image
image_path = "bricklaying_data/frame_0000.png"
image = cv2.imread(image_path)
clone = image.copy()

# Store clicked points
clicked_points = []

# Mouse callback
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        clicked_points.append({"x": x, "y": y})
        cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", clone)

# Show image
cv2.imshow("Image", clone)
cv2.setMouseCallback("Image", click_event)

print("Click to select point(s). Press any key to finish and save.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save to JSON
output_json = "trowel_tip_vertex.json"
with open(output_json, "w") as f:
    json.dump(clicked_points, f, indent=4)

print(f"Saved {len(clicked_points)} point(s) to {output_json}")
