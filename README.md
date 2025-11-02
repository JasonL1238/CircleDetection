# Circle Detection from Different Angles

A computer vision project for detecting circles in images captured from various viewing angles.

## Project Structure

```
circle_detection/
├── src/              # Source code
│   └── detect_circles.py
├── input/            # Place input images here
├── output/           # Processed images with detected circles
├── images/           # Additional image resources
├── results/          # Analysis results and data
├── requirements.txt  # Python dependencies
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your images in the `input/` folder

3. Run the detection script:
```bash
python src/detect_circles.py
```

## Usage

The main detection function is in `src/detect_circles.py`. You can import and use it:

```python
from src.detect_circles import detect_circles

circles, output_img = detect_circles('input/your_image.jpg', 'output/result.jpg')
print(f"Detected {len(circles)} circles")
```

