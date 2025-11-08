# RANSAC + k-NN Hoop Detection Pipeline

A sophisticated computer vision pipeline for detecting red hoops using HSV color masking, edge detection, clustering, RANSAC ellipse fitting, and k-NN selection.

## Pipeline Overview

This pipeline implements the following stages:

1. **HSV Color Masking**: Filters for red regions in HSV color space
2. **Edge Detection**: Extracts Canny edges from the masked region
3. **Clustering (Optional)**: Uses DBSCAN to separate edge points from different hoops
4. **RANSAC Ellipse Fitting**: Fits robust ellipse models to each cluster
5. **k-NN Selection**: Selects the nearest hoop based on distance and/or area

## Project Structure

```
pipelines/ransac_knn/
├── src/
│   └── detect_hoops_ransac_knn.py  # Main detection script
├── output/
│   ├── canny_edges/                 # Canny edge images
│   └── results/                     # Detection results with ellipses drawn
└── README.md
```

## Setup

1. Install dependencies (from project root):
```bash
pip install -r requirements.txt
```

Required packages:
- `opencv-python`
- `numpy`
- `scikit-learn` (for DBSCAN and k-NN)
- `scipy`

## Usage

### Process Video

```bash
cd pipelines/ransac_knn
python src/detect_hoops_ransac_knn.py
```

### Process Individual Frames

```python
from pipelines.ransac_knn.src.detect_hoops_ransac_knn import process_frames

process_frames(
    "input/front_frames",
    "pipelines/ransac_knn/output/results",
    "pipelines/ransac_knn/output/canny_edges",
    num_frames=100
)
```

### Process Single Image

```python
from pipelines.ransac_knn.src.detect_hoops_ransac_knn import detect_hoops_ransac_knn
import cv2

img = cv2.imread("input/frame_000000.jpg")
result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ransac_knn(img)

cv2.imwrite("output.jpg", result_img)
cv2.imwrite("edges.jpg", edge_img)
```

## Parameters

### HSV Masking
- `lower_red1`, `upper_red1`: First red range in HSV (default: (0,50,50) to (10,255,255))
- `lower_red2`, `upper_red2`: Second red range for wrap-around (default: (170,50,50) to (180,255,255))

### Edge Detection
- `canny_low`: Lower Canny threshold (default: 50)
- `canny_high`: Upper Canny threshold (default: 150)

### Clustering
- `use_clustering`: Enable DBSCAN clustering (default: True)
- `cluster_eps`: Maximum distance between points in same cluster (default: 50)
- `cluster_min_samples`: Minimum points to form a cluster (default: 10)

### RANSAC
- `ransac_min_samples`: Minimum points to fit ellipse (default: 5)
- `ransac_max_iterations`: Maximum RANSAC iterations (default: 1000)
- `ransac_distance_threshold`: Distance threshold for inliers (default: 5.0)
- `ransac_min_inliers`: Minimum inliers to accept ellipse (default: 10)

### k-NN Selection
- `reference_point`: (x, y) reference point (default: image center)
- `use_area`: Consider area in selection (default: True)
- `area_weight`: Weight for area feature (default: 0.3)
- `distance_weight`: Weight for distance feature (default: 0.7)

### Visualization
- `draw_all_ellipses`: Draw all detected ellipses (False = only nearest)

## Output

- **Result images**: Original images with detected ellipses drawn
  - Green ellipse: Selected nearest hoop
  - Red circle: Reference point (if specified)
  - Blue ellipses: All detected hoops (if `draw_all_ellipses=True`)

- **Canny edge images**: Binary edge images saved to `output/canny_edges/`

## How It Works

1. **Color Filtering**: Converts image to HSV and creates a binary mask for red regions
2. **Edge Extraction**: Applies Canny edge detection on the masked region
3. **Clustering**: Groups edge points that belong to the same hoop using DBSCAN
4. **RANSAC Fitting**: For each cluster, uses RANSAC to robustly fit an ellipse model
5. **Selection**: Uses k-NN with weighted features (distance + area) to select the nearest hoop

The pipeline is designed to handle multiple hoops in the same frame and automatically selects the most relevant one based on proximity and apparent size.

