# Spreading Mortar Constraint Design

This project provides tools for analyzing and visualizing the 3D geometry of brick wall construction and trowel motion, using depth maps and 2D/3D polygon annotations. It includes robust plane fitting, PCA-based frame estimation, and both static and interactive 3D visualizations.

---

## Features

- **3D Brick Wall Analysis:**  
  - Extracts the side surface of a brick wall from depth maps.
  - Computes a local PCA frame and fits a best-fit plane.
  - Visualizes the wall as a 3D box with axes.

- **Trowel Trajectory Extraction:**  
  - Extracts the trowel’s 3D trajectory from 2D polygons and depth.
  - Computes canonical tip points and local frames per frame.
  - Fits a robust point-on-plane constraint (Huber/LS/Median) to the trowel tip points.

- **Visualization:**  
  - Static and interactive 3D visualization using PyVista.
  - Shows wall, trowel triangles, tip trajectory, fitted plane, and residuals.

- **Trajectory Modeling (ProMP):**  
  - [unit_test_4.py](unit_test_4.py): Learns a Probabilistic Movement Primitive (ProMP) from a single demonstration and generates new trajectories.

---

## Requirements

- Python 3.8+
- numpy
- opencv-python
- pyvista
- matplotlib
- tqdm
- torch
- scipy

Install all dependencies:
```sh
pip install -r requirements.txt
```

---

## Data Files

Place the following files in the project root:
- `depth_map_cross_frames_refined.npy` – Depth maps for all frames (N, H, W)
- `trowel_polygon_vertices.npy` – 2D trowel polygon vertices per frame
- `brick_wall_side_surface.npy` – 2D polygon for the brick wall side surface

Some scripts may require:
- `trowel_tip_polygon_vertices.npy` – (optional) 2D tip vertices
- `T_trowel2cam.npy`, `T_wall2cam.npy` – (optional) Precomputed pose matrices

---

## Usage

### Main 3D Analysis & Visualization

Run:
```sh
python unit_test_6.py
```
- Estimates the wall’s top plane using PCA.
- Extracts trowel tip points in 3D.
- Fits a robust point-on-plane constraint.
- Prints residual statistics.
- Opens a PyVista 3D visualization.

### Other Visualizations

- `unit_test_3.py`: Static 3D visualization of wall and trowel trajectory.
- `unit_test_4.py`: ProMP learning and trajectory generation.
- `unit_test_5.py`: Computes and saves trowel and wall pose matrices.

---

## DepthAnythingV2 Model

Some utilities/scripts may require the DepthAnythingV2 model weights.  
Download from: https://github.com/DepthAnything/Depth-Anything-V2  
Choose: **Depth-Anything-V2-Large 335.3M**

---

## Utilities

- `utils/project_to_3D.py`: Animated 3D keypoint trajectories.
- `utils/project_to_2D.py`: Animated 2D projections.
- `utils/interactive_segment.py`: Interactive segmentation for mask creation.
- `utils/video2img.py`: Extracts frames from video.

---

## Example Output

- **Console:**  
  - Plane fit and residual statistics (mean, max, std in mm).
- **PyVista:**  
  - 3D scene with wall, trowel, fitted plane, tip points, and residuals.

---

## Citation

If you use this code or data for research, please
