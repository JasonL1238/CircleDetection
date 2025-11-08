# Circle Detection from Different Angles

A computer vision project for detecting circles/hoops in images captured from various viewing angles.

## Project Structure

```
circle_detection/
├── input/                    # Input data
│   ├── front_frames/         # Individual frame images
│   └── front.mp4            # Video file
├── pipelines/                # Detection pipelines
│   ├── canny_edges/         # Simple Canny edge detection
│   │   ├── src/
│   │   ├── output/
│   │   └── README.md
│   └── ransac_knn/          # RANSAC + k-NN hoop detection
│       ├── src/
│       ├── output/
│       └── README.md
├── requirements.txt          # Python dependencies
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your input data in the `input/` folder:
   - Video files: `input/front.mp4`
   - Frame images: `input/front_frames/frame_XXXXXX.jpg`

## Available Pipelines

### 1. Canny Edge Detection (`pipelines/canny_edges/`)

Simple edge detection pipeline that extracts Canny edges from video frames.

**Usage:**
```bash
cd pipelines/canny_edges
python src/detect_canny_edges.py
```

See [pipelines/canny_edges/README.md](pipelines/canny_edges/README.md) for details.

### 2. RANSAC + k-NN Hoop Detection (`pipelines/ransac_knn/`)

Advanced pipeline for detecting red hoops using:
- HSV color masking
- Edge detection
- DBSCAN clustering
- RANSAC ellipse fitting
- k-NN selection

**Usage:**
```bash
cd pipelines/ransac_knn
python src/detect_hoops_ransac_knn.py
```

See [pipelines/ransac_knn/README.md](pipelines/ransac_knn/README.md) for details.

## Output Structure

Each pipeline has its own output directory:
- `pipelines/canny_edges/output/canny_edges/` - Edge images
- `pipelines/ransac_knn/output/canny_edges/` - Edge images
- `pipelines/ransac_knn/output/results/` - Detection results with ellipses

## Dependencies

- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `scipy>=1.10.0`
- `scikit-image>=0.21.0`
- `scikit-learn>=1.3.0` (for RANSAC + k-NN pipeline)
