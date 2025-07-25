
"""
    After setting the sampling region, sample candidate keypoints from the region

"""



import random
import numpy as np
from shapely.geometry import Polygon, Point
import json

# Your manually selected points from JSON file

with open("selected_keypoints.json", "r") as f:
    data = json.load(f)

# Convert to list of (x, y) tuples
polygon_points = [(p["x"], p["y"]) for p in data]


# Create polygon
poly = Polygon(polygon_points)
min_x, min_y, max_x, max_y = poly.bounds

# Uniformly sample N points within the bounding box and test if in polygon
N = 10
sampled_points = []
while len(sampled_points) < N:
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    if poly.contains(Point(x, y)):
        sampled_points.append((x, y))



# Optional: visualize
import matplotlib.pyplot as plt

px, py = zip(*polygon_points)
sx, sy = zip(*sampled_points)

plt.figure(figsize=(6, 6))
plt.plot(px + (px[0],), py + (py[0],), 'b-', label='Polygon')
plt.scatter(sx, sy, color='red', s=10, label='Sampled Points')
plt.legend()
plt.axis('equal')
plt.show()



# Store in JSON
# Store all points in a flat list
output_json = "sampling_candidate_keypoints.json"
all_points = [{"x": int(x), "y": int(y)} for (x, y) in polygon_points + sampled_points]

with open(output_json, "w") as f:
    json.dump(all_points, f, indent=2)
