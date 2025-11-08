# Hough Transform for Ellipses Hoop Detection Pipeline

A computer vision pipeline for detecting red hoops using Hough Transform for Ellipses from scikit-image. This method is robust to partial edges and naturally handles multiple rings, making it ideal for cases where the camera view changes or partial ellipses appear.

## Pipeline Overview

This pipeline implements the following stages:

1. **HSV Color Masking**: Filters for red regions in HSV color space
2. **Canny Edge Detection**: Extracts edges using scikit-image's Canny edge detector
3. **Hough Transform for Ellipses**: Uses `skimage.transform.hough_ellipse` to detect ellipses from edge images
4. **Selection**: Selects the nearest hoop based on distance from reference point

## Project Structure

```
pipelines/hough_ellipse/
├── src/
│   └── detect_hoops_hough_ellipse.py  # Main detection script
├── output/
│   ├── test_canny_edges/              # Canny edge images
│   └── test_results/                  # Detection results with ellipses drawn
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
- `scikit-image` (for `hough_ellipse` and `canny`)
- `scipy`

## Usage

### Process Test Images

```bash
cd pipelines/hough_ellipse
python src/detect_hoops_hough_ellipse.py
```

### Process Video

```python
from pipelines.hough_ellipse.src.detect_hoops_hough_ellipse import process_video

process_video(
    "input/front.mp4",
    "pipelines/hough_ellipse/output/results",
    "pipelines/hough_ellipse/output/canny_edges"
)
```

### Process Individual Frames

```python
from pipelines.hough_ellipse.src.detect_hoops_hough_ellipse import process_frames

process_frames(
    "input/front_frames",
    "pipelines/hough_ellipse/output/results",
    "pipelines/hough_ellipse/output/canny_edges",
    num_frames=100
)
```

### Process Single Image

```python
from pipelines.hough_ellipse.src.detect_hoops_hough_ellipse import detect_hoops_hough_ellipse
import cv2

img = cv2.imread("input/frame_000000.jpg")
result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_hough_ellipse(img)

cv2.imwrite("output.jpg", result_img)
cv2.imwrite("edges.jpg", edge_img)
```

## Parameters

### HSV Masking
- `use_red_mask`: Enable red color masking (default: True)
- `lower_red1`, `upper_red1`: First red range in HSV (default: (0,50,50) to (10,255,255))
- `lower_red2`, `upper_red2`: Second red range for wrap-around (default: (170,50,50) to (180,255,255))

### Canny Edge Detection
- `canny_sigma`: Sigma parameter for Canny edge detection (default: 2.0)
  - Higher values = smoother edges, fewer details
  - Lower values = more detailed edges, more noise

### Hough Ellipse Parameters
- `accuracy`: Accuracy parameter for hough_ellipse (default: 20)
  - Higher values = more accurate but slower
  - Lower values = faster but less accurate
- `threshold`: Threshold parameter for hough_ellipse (default: 50)
  - Higher values = fewer false positives, may miss some ellipses
  - Lower values = more detections, may include false positives
- `min_size`: Minimum ellipse size in pixels (default: 30)
- `max_size`: Maximum ellipse size in pixels (default: 200)

### Selection
- `reference_point`: (x, y) reference point for selection (default: image center)
- `draw_all_ellipses`: Draw all detected ellipses (False = only nearest)

## Output

- **Result images**: Original images with detected ellipses drawn
  - Green ellipse: Selected nearest hoop
  - Blue circle: Detected center point
  - Red circle: Reference point (if specified)
  - Blue ellipses: All detected hoops (if `draw_all_ellipses=True`)

- **Canny edge images**: Binary edge images saved to `output/test_canny_edges/`

## How It Works

1. **Color Filtering**: Converts image to HSV and creates a binary mask for red regions
2. **Edge Extraction**: Applies Canny edge detection using scikit-image (with optional masking)
3. **Hough Transform**: Uses `hough_ellipse` to detect ellipses from edge points
   - The algorithm accumulates votes in parameter space for ellipse candidates
   - Results are sorted by accumulator value (confidence)
4. **Selection**: Selects the nearest ellipse based on distance from reference point

## Advantages

✅ **Robust to partial edges**: Hough Transform can detect ellipses even when edges are incomplete
✅ **Naturally handles multiple rings**: Can detect multiple ellipses in a single frame
✅ **Works with changing camera views**: Handles different viewing angles and partial ellipses

## Limitations

❌ **Slower processing**: Typically takes hundreds of milliseconds per frame, not ideal for real-time applications
❌ **Parameter tuning**: Requires careful tuning of `accuracy`, `threshold`, `min_size`, and `max_size` for optimal results

## Performance Notes

- Processing time: ~200-500ms per frame (depending on image size and parameters)
- Best for: Offline processing, batch processing, or when robustness is more important than speed
- Not recommended for: Real-time video processing where speed is critical

## Comparison with Other Pipelines

- **vs. RANSAC + k-NN**: Hough Transform is more robust to partial edges but slower
- **vs. Canny Edges only**: Hough Transform actually detects and fits ellipses, not just edges
- **vs. OpenCV HoughCircles**: Hough Ellipse handles non-circular ellipses and different orientations

