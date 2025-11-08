"""
Hough Transform for Ellipses Hoop Detection Pipeline
Uses scikit-image's hough_ellipse for robust ellipse detection from edge images
"""

import cv2
import numpy as np
from pathlib import Path
from skimage import feature, transform, color
import math


def make_red_mask(hsv_img, lower_red1=(0, 50, 50), upper_red1=(10, 255, 255),
                  lower_red2=(170, 50, 50), upper_red2=(180, 255, 255)):
    """
    Create a mask for red color in HSV space
    Red wraps around 180, so we need two ranges
    
    Args:
        hsv_img: Image in HSV color space
        lower_red1: Lower bound for red range 1 (H, S, V)
        upper_red1: Upper bound for red range 1
        lower_red2: Lower bound for red range 2 (wraps around)
        upper_red2: Upper bound for red range 2
    
    Returns:
        Binary mask where red regions are white
    """
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    
    # Combine both masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask


def detect_hoops_hough_ellipse(img,
                               # HSV masking parameters
                               use_red_mask=True,
                               lower_red1=(0, 50, 50), upper_red1=(10, 255, 255),
                               lower_red2=(170, 50, 50), upper_red2=(180, 255, 255),
                               # Canny edge detection parameters
                               canny_sigma=2.0,
                               # Hough ellipse parameters
                               accuracy=20,
                               threshold=50,
                               min_size=30,
                               max_size=200,
                               # Selection parameters
                               reference_point=None,
                               draw_all_ellipses=False):
    """
    Detect hoops using Hough Transform for Ellipses (scikit-image)
    
    Args:
        img: Input BGR image
        use_red_mask: Whether to use red color masking
        lower_red1, upper_red1: First red range in HSV
        lower_red2, upper_red2: Second red range in HSV (wrap-around)
        canny_sigma: Sigma parameter for Canny edge detection
        accuracy: Accuracy parameter for hough_ellipse (default: 20)
        threshold: Threshold parameter for hough_ellipse (default: 50)
        min_size: Minimum ellipse size (default: 30)
        max_size: Maximum ellipse size (default: 200)
        reference_point: (x, y) reference point for selection (default: image center)
        draw_all_ellipses: If True, draw all detected ellipses; if False, only nearest
    
    Returns:
        (result_image, detected_ellipses, nearest_ellipse_idx, edge_image, mask)
    """
    h, w = img.shape[:2]
    
    # Step 1: Create red mask if enabled
    mask = None
    if use_red_mask:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = make_red_mask(hsv, lower_red1, upper_red1, lower_red2, upper_red2)
    
    # Step 2: Convert to grayscale and apply mask if available
    # Convert BGR to RGB first, then to grayscale
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Ensure image is in 0-1 range for scikit-image
    if rgb_img.dtype == np.uint8:
        rgb_img = rgb_img.astype(np.float32) / 255.0
    
    gray = color.rgb2gray(rgb_img)
    
    if mask is not None:
        # Apply mask to grayscale image (normalize mask to 0-1)
        mask_normalized = (mask > 0).astype(np.float32)
        gray = gray * mask_normalized
    
    # Step 3: Apply Canny edge detection using scikit-image
    edges = feature.canny(gray, sigma=canny_sigma)
    
    # Step 4: Apply Hough Transform for Ellipses
    try:
        result = transform.hough_ellipse(
            edges,
            accuracy=accuracy,
            threshold=threshold,
            min_size=min_size,
            max_size=max_size
        )
        
        # Sort by accumulator value (higher is better)
        if len(result) > 0:
            result.sort(order='accumulator')
            detected_ellipses = []
            
            for ellipse_result in result:
                yc = ellipse_result['yc']
                xc = ellipse_result['xc']
                a = ellipse_result['a']  # Semi-major axis
                b = ellipse_result['b']  # Semi-minor axis
                orientation = ellipse_result['orientation']
                
                # Convert to OpenCV format: (center, axes, angle)
                # OpenCV uses (width, height) for axes, and angle in degrees
                center = (int(xc), int(yc))
                axes = (int(a * 2), int(b * 2))  # Convert semi-axes to full axes
                angle = math.degrees(orientation)
                
                # Sanity checks
                if (0 <= center[0] < w and 0 <= center[1] < h and
                    axes[0] > 0 and axes[1] > 0 and
                    axes[0] < w * 2 and axes[1] < h * 2):
                    detected_ellipses.append({
                        'center': center,
                        'axes': axes,
                        'angle': angle,
                        'xc': xc,
                        'yc': yc,
                        'a': a,
                        'b': b,
                        'orientation': orientation,
                        'accumulator': ellipse_result['accumulator']
                    })
        else:
            detected_ellipses = []
    
    except Exception as e:
        print(f"  Warning: Hough ellipse detection failed: {e}")
        detected_ellipses = []
    
    # Step 5: Select nearest ellipse if multiple detected
    nearest_idx = None
    if len(detected_ellipses) > 0:
        if reference_point is None:
            reference_point = (w // 2, h // 2)
        
        # Calculate distances from reference point
        distances = []
        for ellipse in detected_ellipses:
            cx, cy = ellipse['center']
            dist = math.sqrt((cx - reference_point[0])**2 + (cy - reference_point[1])**2)
            distances.append(dist)
        
        nearest_idx = np.argmin(distances)
    
    # Step 6: Draw results
    result_img = img.copy()
    
    if draw_all_ellipses:
        # Draw all detected ellipses in blue
        for i, ellipse in enumerate(detected_ellipses):
            center = ellipse['center']
            axes = ellipse['axes']
            angle = ellipse['angle']
            cv2.ellipse(result_img, center, axes, angle, 0, 360, (255, 0, 0), 2)
            cv2.circle(result_img, center, 5, (255, 0, 0), -1)
    else:
        # Draw only nearest ellipse in green
        if nearest_idx is not None:
            ellipse = detected_ellipses[nearest_idx]
            center = ellipse['center']
            axes = ellipse['axes']
            angle = ellipse['angle']
            cv2.ellipse(result_img, center, axes, angle, 0, 360, (0, 255, 0), 3)
            cv2.circle(result_img, center, 5, (0, 255, 0), -1)
            
            # Draw big blue dot at determined center
            cv2.circle(result_img, center, 20, (255, 0, 0), -1)  # Big blue filled circle
            
            # Draw reference point
            if reference_point:
                cv2.circle(result_img, reference_point, 5, (0, 0, 255), -1)
    
    # Convert edge image to uint8 for saving
    edge_image = (edges * 255).astype(np.uint8)
    
    return result_img, detected_ellipses, nearest_idx, edge_image, mask


def process_video(video_path, output_dir, canny_edges_dir, **kwargs):
    """
    Process video file with Hough Ellipse pipeline
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        **kwargs: Additional parameters for detect_hoops_hough_ellipse
    """
    video_path = Path(video_path)
    output_path = Path(output_dir)
    canny_path = Path(canny_edges_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    canny_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video has {total_frames} frames")
    
    processed = 0
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hoops
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_hough_ellipse(
            frame, **kwargs
        )
        
        # Save result image
        output_file = output_path / f"frame_{frame_num:06d}_result.jpg"
        cv2.imwrite(str(output_file), result_img)
        
        # Save Canny edge image
        canny_file = canny_path / f"frame_{frame_num:06d}_edges.jpg"
        cv2.imwrite(str(canny_file), edge_img)
        
        processed += 1
        frame_num += 1
        
        if processed % 10 == 0:
            print(f"  Processed {processed}/{total_frames} frames...")
            if len(ellipses) > 0:
                print(f"    Detected {len(ellipses)} ellipse(s), nearest: {nearest_idx}")
    
    cap.release()
    print(f"\n✓ Processed {processed} frames")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def process_frames(input_dir, output_dir, canny_edges_dir, num_frames=None, **kwargs):
    """
    Process frames with Hough Ellipse pipeline
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        num_frames: Number of frames to process (None = process all available)
        **kwargs: Additional parameters for detect_hoops_hough_ellipse
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    canny_path = Path(canny_edges_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    canny_path.mkdir(parents=True, exist_ok=True)
    
    # Find available frames
    if num_frames is None:
        frame_files = sorted(input_path.glob("frame_*.jpg"))
        if frame_files:
            max_frame = 0
            for frame_file in frame_files:
                try:
                    frame_num = int(frame_file.stem.split('_')[1])
                    max_frame = max(max_frame, frame_num)
                except:
                    pass
            num_frames = max_frame + 1
            print(f"  Found {num_frames} frames to process")
        else:
            print("  No frames found!")
            return
    
    processed = 0
    for i in range(num_frames):
        frame_name = f"frame_{i:06d}.jpg"
        frame_path = input_path / frame_name
        
        if not frame_path.exists():
            if processed > 0:
                break
            continue
        
        # Load image
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  Could not load {frame_name}, skipping...")
            continue
        
        # Detect hoops
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_hough_ellipse(
            img, **kwargs
        )
        
        # Save result image
        output_file = output_path / f"{frame_name.replace('.jpg', '_result.jpg')}"
        cv2.imwrite(str(output_file), result_img)
        
        # Save Canny edge image
        canny_file = canny_path / f"{frame_name.replace('.jpg', '_edges.jpg')}"
        cv2.imwrite(str(canny_file), edge_img)
        
        processed += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
            if len(ellipses) > 0:
                print(f"    Detected {len(ellipses)} ellipse(s), nearest: {nearest_idx}")
    
    print(f"\n✓ Processed {processed} frames")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def process_test_images(test_images_dir, output_dir, canny_edges_dir, **kwargs):
    """
    Process test images with Hough Ellipse pipeline
    
    Args:
        test_images_dir: Directory containing test images
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        **kwargs: Additional parameters for detect_hoops_hough_ellipse
    """
    test_path = Path(test_images_dir)
    output_path = Path(output_dir)
    canny_path = Path(canny_edges_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    canny_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(test_path.glob(ext))
        image_files.extend(test_path.glob(ext.upper()))
    
    if not image_files:
        print(f"Error: No test images found in {test_path}")
        return
    
    print(f"Found {len(image_files)} test images")
    
    processed = 0
    for img_file in sorted(image_files):
        print(f"\nProcessing: {img_file.name}")
        
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  Could not load {img_file.name}, skipping...")
            continue
        
        # Detect hoops
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_hough_ellipse(
            img, **kwargs
        )
        
        # Save result image
        output_file = output_path / f"{img_file.stem}_result.jpg"
        cv2.imwrite(str(output_file), result_img)
        
        # Save Canny edge image
        canny_file = canny_path / f"{img_file.stem}_edges.jpg"
        cv2.imwrite(str(canny_file), edge_img)
        
        # Print detection info
        if len(ellipses) > 0:
            print(f"  ✓ Detected {len(ellipses)} ellipse(s)")
            if nearest_idx is not None:
                ellipse = ellipses[nearest_idx]
                center = ellipse['center']
                axes = ellipse['axes']
                angle = ellipse['angle']
                print(f"    Selected ellipse: center={center}, axes={axes}, angle={angle:.1f}°")
        else:
            print(f"  ✗ No ellipses detected")
        
        processed += 1
    
    print(f"\n✓ Processed {processed} test images")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def main():
    """Process test images with Hough Ellipse pipeline"""
    test_images_dir = Path(__file__).parent.parent.parent.parent / "input" / "test_images"
    output_dir = Path(__file__).parent.parent / "output" / "test_results"
    canny_edges_dir = Path(__file__).parent.parent / "output" / "test_canny_edges"
    
    print(f"Processing test images with Hough Ellipse pipeline...")
    print(f"Input: {test_images_dir}")
    print(f"Results: {output_dir}")
    print(f"Canny edges: {canny_edges_dir}\n")
    
    process_test_images(
        test_images_dir,
        output_dir,
        canny_edges_dir,
        use_red_mask=True,  # Use red masking
        canny_sigma=2.0,  # Canny edge detection sigma
        accuracy=20,  # Hough ellipse accuracy
        threshold=50,  # Hough ellipse threshold
        min_size=30,  # Minimum ellipse size
        max_size=200,  # Maximum ellipse size
        draw_all_ellipses=False,  # Only draw nearest ellipse
    )


if __name__ == "__main__":
    main()

