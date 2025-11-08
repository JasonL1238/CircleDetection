# Partial Ellipse Fit Hoop Detection Pipeline

A robust computer vision pipeline for detecting hoops in challenging scenarios: partially visible hoops, close-range views, and broken arcs. Uses direct least-squares ellipse fitting (Fitzgibbon's method) that works even with as little as 25% of the ellipse visible.

## Pipeline Overview

This pipeline is specifically designed to handle the two hardest geometric failure modes:

1. **Partially visible hoops** (maybe only the top half)
2. **Very close hoops** where you don't see nested ellipses - just one big, broken arc

### Key Features

- **Direct Least-Squares Ellipse Fitting**: Uses Fitzgibbon's method to fit ellipses to partial arcs (works with as little as 25% of the ellipse)
- **Arc Clustering**: Uses DBSCAN to cluster edge points into spatially coherent arc segments
- **Circle Fallback**: Automatically falls back to circle fitting when ellipse fitting fails
- **Inner Edge Focus**: Optionally focuses on inner edges only for tighter contours
- **Robust Selection**: Selects best ellipse based on size, gradient density, and proximity

## Project Structure

```
pipelines/partial_ellipse_fit/
├── src/
│   └── detect_hoops_partial_ellipse.py  # Main detection script
├── output/
│   ├── test_canny_edges/                # Canny edge images
│   └── test_results/                    # Detection results with ellipses drawn
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
- `scikit-learn` (for DBSCAN clustering)

## Usage

### Process Test Images

```bash
cd pipelines/partial_ellipse_fit
python src/detect_hoops_partial_ellipse.py
```

### Process Video

```python
from pipelines.partial_ellipse_fit.src.detect_hoops_partial_ellipse import process_video

process_video(
    "input/front.mp4",
    "pipelines/partial_ellipse_fit/output/results",
    "pipelines/partial_ellipse_fit/output/canny_edges"
)
```

### Process Single Image

```python
from pipelines.partial_ellipse_fit.src.detect_hoops_partial_ellipse import detect_hoops_partial_ellipse
import cv2

img = cv2.imread("input/frame_000000.jpg")
result_img, ellipses, best_idx, edge_img, mask = detect_hoops_partial_ellipse(img)

cv2.imwrite("output.jpg", result_img)
cv2.imwrite("edges.jpg", edge_img)
```

## Parameters

### HSV Masking
- `use_red_mask`: Enable red color masking (default: True)
- `lower_red1`, `upper_red1`: First red range in HSV (default: (0,50,50) to (10,255,255))
- `lower_red2`, `upper_red2`: Second red range for wrap-around (default: (170,50,50) to (180,255,255))

### Edge Detection
- `canny_low`: Lower Canny threshold (default: 50)
- `canny_high`: Upper Canny threshold (default: 150)
- `use_inner_edges`: Focus on inner edges only (default: True)

### Clustering
- `cluster_eps`: Maximum distance between points in same cluster (default: 10)
- `cluster_min_samples`: Minimum points to form a cluster (default: 20)

### Selection
- `reference_point`: (x, y) reference point for selection (default: image center)
- `prefer_largest`: Prefer ellipses with larger minor axis (default: True)
- `prefer_highest_gradient`: Prefer ellipses with higher gradient density (default: True)

### Fallback
- `use_circle_fallback`: Fallback to circle fitting if ellipse fails (default: True)
- `min_points_for_ellipse`: Minimum points required for ellipse fitting (default: 5)

### Visualization
- `draw_all_ellipses`: Draw all detected ellipses (False = only best)

## How It Works

### 1. Edge Point Extraction
- Applies Canny edge detection (optionally focusing on inner edges only)
- Extracts all edge point coordinates

### 2. Arc Clustering
- Uses DBSCAN to cluster edge points into spatially coherent groups
- Each cluster represents a potential hoop arc segment

### 3. Direct Ellipse Fitting
- For each cluster, applies Fitzgibbon's Direct Least Squares method
- This method works even with partial arcs (as little as 25% of the ellipse)
- Converts conic coefficients to standard ellipse parameters (center, axes, angle)

### 4. Circle Fallback
- If ellipse fitting fails, falls back to `cv2.minEnclosingCircle`
- Useful for very close hoops that appear almost circular

### 5. Best Ellipse Selection
- Scores each detected ellipse based on:
  - Size (larger minor axis preferred)
  - Gradient density (stronger edges preferred)
  - Proximity to reference point
- Selects the highest-scoring ellipse

## Advantages

✅ **Works with Partial Arcs**: Can detect hoops with as little as 25% visibility
✅ **Handles Close-Range**: Circle fallback for very close hoops
✅ **Robust to Noise**: DBSCAN clustering isolates coherent arc segments
✅ **No Full Ellipse Required**: Doesn't need complete, nested ellipses
✅ **Tight Contours**: Focuses on inner edges for accurate fitting

## Limitations

❌ **Slower than Hough Transform**: More computationally intensive due to clustering and fitting
❌ **Parameter Tuning**: Requires tuning of clustering parameters for different scenarios
❌ **Multiple Arcs**: May detect multiple ellipses from the same hoop if arcs are separated

## Comparison with Other Pipelines

- **vs. RANSAC + k-NN**: Better for partial hoops, but slower
- **vs. Hough Ellipse**: Works with much smaller arc segments, but more complex
- **vs. Ellipse-YOLO**: Traditional CV approach, no training required, but less accurate than deep learning

## Use Cases

This pipeline is ideal for:

- **Partially occluded hoops** (only top/bottom half visible)
- **Close-range views** where hoops appear as broken arcs
- **Motion blur** scenarios where edges are fragmented
- **Low contrast** situations where full ellipse detection fails

## Algorithm Details

### Fitzgibbon's Direct Least Squares Method

The algorithm solves the generalized eigenvalue problem:
```
S * v = λ * C * v
```

Where:
- `S` is the scatter matrix from edge points
- `C` is the constraint matrix ensuring ellipse shape (4ac - b² > 0)
- `v` is the eigenvector containing conic coefficients

This directly fits an ellipse to the points without requiring iterative optimization, making it fast and robust to partial arcs.

