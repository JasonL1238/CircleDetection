# Canny Edge Detection Pipeline

Simple Canny edge detection pipeline for extracting edges from video frames.

## Pipeline Overview

This pipeline performs basic edge detection:
1. Load video frames or images
2. Convert to grayscale
3. Apply Gaussian blur
4. Apply Canny edge detection
5. Save edge images

## Project Structure

```
pipelines/canny_edges/
├── src/
│   ├── detect_canny_edges.py  # Main edge detection script
│   └── create_edge_video.py   # Create video from edge images
├── output/
│   └── canny_edges/            # Canny edge images
└── README.md
```

## Usage

### Process Video

```bash
cd pipelines/canny_edges
python src/detect_canny_edges.py
```

### Create Video from Edge Images

```bash
python src/create_edge_video.py
```

### Use as Module

```python
from pipelines.canny_edges.src.detect_canny_edges import process_video, process_frames
from pathlib import Path

# Process video
process_video(
    "input/front.mp4",
    "pipelines/canny_edges/output/canny_edges",
    canny_low=50,
    canny_high=150
)

# Process frames
process_frames(
    "input/front_frames",
    "pipelines/canny_edges/output/canny_edges",
    num_frames=100
)
```

## Parameters

- `canny_low`: Lower threshold for Canny edge detection (default: 50)
- `canny_high`: Upper threshold for Canny edge detection (default: 150)

## Output

Edge images are saved to `output/canny_edges/` with naming pattern:
- `frame_XXXXXX_edges.jpg` for video processing
- `frame_XXXXXX_edges.jpg` for frame processing

