# Ellipse-YOLO Hoop Detection Pipeline

An end-to-end deep learning pipeline for detecting circular markers' centers using rotated bounding boxes. This pipeline is based on the **Ellipse-YOLO** model described in the paper: "Ellipse-YOLO: A Near Real-Time Detection of Circular Marks' Centers Network Based on High-Speed Photogrammetry Images".

## Pipeline Overview

This pipeline implements an end-to-end deep learning approach that:

1. **Uses Rotated Bounding Boxes**: Unlike traditional horizontal detectors, Ellipse-YOLO predicts rotated bounding boxes that tightly fit circular markers
2. **Direct Center Extraction**: Extracts marker centers directly from the detection boxes without requiring additional post-processing
3. **Near Real-Time Performance**: Achieves high accuracy (83.5% AP) with fast inference (41.2 FPS)
4. **Robust to Various Conditions**: Handles varying lighting, angles, occlusions, and partial ellipses

### Key Features

- **Enhanced Angle Encoding**: Uses Angle Continuous Mapping (ACM) to handle angle prediction discontinuities
- **Multi-Scale Oriented Fusion Attention (MOA)**: Captures orientation-aware features and multi-scale information
- **Improved Small Target Detection**: Enhanced feature fusion for detecting small circular markers
- **End-to-End**: Directly outputs ellipse parameters (center, axes, angle) without additional algorithms

## Project Structure

```
pipelines/ellipse_yolo/
├── src/
│   └── detect_hoops_ellipse_yolo.py  # Main detection script
├── models/                            # Place trained model here (optional)
│   └── ellipse_yolo.pt                # Model weights file
├── output/
│   ├── test_canny_edges/              # Canny edge images
│   └── test_results/                  # Detection results with ellipses drawn
└── README.md
```

## Setup

### 1. Install Dependencies

From project root:
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch (Required for Model)

If you plan to use a trained Ellipse-YOLO model, install PyTorch:

```bash
# For CPU only
pip install torch torchvision

# For GPU support (CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Model Integration

**Note**: This pipeline includes a framework for loading and using a trained Ellipse-YOLO model. However, you need to:

1. **Train or obtain a trained model**: The model architecture should match the Ellipse-YOLO paper specifications
2. **Place model weights**: Save your trained model as `ellipse_yolo.pt` in the `models/` directory
3. **Update model loading code**: Modify `load_ellipse_yolo_model()` to match your actual model architecture

**Fallback Method**: If no model is available, the pipeline automatically falls back to OpenCV's `HoughCircles` method for testing purposes.

## Usage

### Process Test Images

```bash
cd pipelines/ellipse_yolo
python src/detect_hoops_ellipse_yolo.py
```

### Process Video

```python
from pipelines.ellipse_yolo.src.detect_hoops_ellipse_yolo import process_video

process_video(
    "input/front.mp4",
    "pipelines/ellipse_yolo/output/results",
    "pipelines/ellipse_yolo/output/canny_edges",
    model_path="pipelines/ellipse_yolo/models/ellipse_yolo.pt"  # Optional
)
```

### Process Individual Frames

```python
from pipelines.ellipse_yolo.src.detect_hoops_ellipse_yolo import process_frames

process_frames(
    "input/front_frames",
    "pipelines/ellipse_yolo/output/results",
    "pipelines/ellipse_yolo/output/canny_edges",
    num_frames=100,
    model_path="path/to/model.pt"  # Optional
)
```

### Process Single Image

```python
from pipelines.ellipse_yolo.src.detect_hoops_ellipse_yolo import detect_hoops_ellipse_yolo
import cv2

img = cv2.imread("input/frame_000000.jpg")
result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ellipse_yolo(
    img,
    model_path="path/to/model.pt"  # Optional
)

cv2.imwrite("output.jpg", result_img)
cv2.imwrite("edges.jpg", edge_img)
```

## Parameters

### Model Parameters
- `model_path`: Path to trained model weights file (.pt or .pth)
- `use_model`: Whether to attempt using the model (default: True)
- `conf_threshold`: Confidence threshold for model predictions (default: 0.25)
- `iou_threshold`: IoU threshold for Non-Maximum Suppression (default: 0.45)
- `img_size`: Input image size for model, should match training (default: 640)

### HSV Masking
- `use_red_mask`: Enable red color masking (default: True)
- `lower_red1`, `upper_red1`: First red range in HSV (default: (0,50,50) to (10,255,255))
- `lower_red2`, `upper_red2`: Second red range for wrap-around (default: (170,50,50) to (180,255,255))

### Fallback Method (HoughCircles)
Used when model is not available:
- `fallback_dp`: Inverse ratio of accumulator resolution (default: 1)
- `fallback_minDist`: Minimum distance between circle centers (default: 30)
- `fallback_param1`: Upper threshold for edge detection (default: 50)
- `fallback_param2`: Accumulator threshold (default: 30)
- `fallback_minRadius`: Minimum circle radius (default: 10)
- `fallback_maxRadius`: Maximum circle radius (default: 100)

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

## Model Architecture (From Paper)

The Ellipse-YOLO model is based on YOLOv8 with three key improvements:

1. **Enhanced Angle Encoding Prediction Head**
   - Uses Angle Continuous Mapping (ACM) to solve angle encoding discontinuities
   - Employs ProbIoU loss function for Gaussian bounding boxes
   - Predicts angle encoding instead of direct angles

2. **Multi-Scale Oriented Fusion Attention (MOA) Module**
   - Three parallel pathways for feature extraction
   - Coordinate-aware branches for spatial information
   - Multi-scale perception branch with large kernel convolutions
   - Cross-spatial information aggregation

3. **Improved Small Target Detection Module**
   - Additional detection layer for small targets
   - Simplified PAN connections
   - Better integration of shallow and deep features

## Performance (From Paper)

- **Detection Accuracy**: 83.5% AP
- **Center Extraction MAE**: 0.019 pixels
- **Inference Speed**: 41.2 FPS (near real-time)
- **Computational Complexity**: 26.8 GFLOPs

## Advantages

✅ **End-to-End**: Directly extracts centers without additional algorithms
✅ **High Accuracy**: 83.5% AP with sub-pixel center accuracy
✅ **Fast**: Near real-time performance (41.2 FPS)
✅ **Robust**: Handles occlusions, rotations, and varying lighting
✅ **Angle-Aware**: Rotated bounding boxes fit circular markers tightly

## Limitations

❌ **Requires Training**: Needs a trained model (not included)
❌ **Model Architecture**: Must match paper specifications
❌ **Dataset**: Requires labeled training data with rotated bounding boxes
❌ **GPU Recommended**: For best performance, GPU acceleration is recommended

## Comparison with Other Pipelines

- **vs. RANSAC + k-NN**: Ellipse-YOLO is faster and more accurate, but requires training
- **vs. Hough Ellipse**: Ellipse-YOLO is much faster (41.2 FPS vs ~200-500ms/frame) and more accurate
- **vs. Traditional Methods**: End-to-end approach eliminates need for multiple algorithm stages

## Model Training (If Implementing)

To train your own Ellipse-YOLO model, you would need to:

1. **Prepare Dataset**: Images with rotated bounding box annotations
2. **Implement Model Architecture**: Based on YOLOv8 with the three key improvements
3. **Train Model**: Using PyTorch with appropriate loss functions (ProbIoU + ACM loss)
4. **Evaluate**: Test on validation set and tune hyperparameters

Refer to the original paper for detailed architecture and training procedures.

## Notes

- The current implementation includes a framework for model loading but requires the actual trained model
- If no model is available, the pipeline automatically uses OpenCV's HoughCircles as a fallback
- To fully utilize this pipeline, you need to either:
  1. Obtain a pre-trained Ellipse-YOLO model
  2. Train your own model following the paper's specifications
  3. Adapt the code to work with a similar rotated object detection model

