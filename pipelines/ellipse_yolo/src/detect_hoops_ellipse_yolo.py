"""
Ellipse-YOLO Hoop Detection Pipeline
End-to-end deep learning model for detecting circular markers' centers using rotated bounding boxes
Based on: "Ellipse-YOLO: A Near Real-Time Detection of Circular Marks' Centers Network"
"""

import cv2
import numpy as np
from pathlib import Path
import math
import warnings

# Try to import PyTorch - if not available, will use fallback method
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Using fallback detection method.")


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


def decode_angle_encoding(encoded_angle, omega=2):
    """
    Decode angle from ACM (Angle Continuous Mapping) encoding
    
    Args:
        encoded_angle: Encoded angle value
        omega: Angular frequency (default: 2)
    
    Returns:
        Decoded angle in radians
    """
    # Simplified decoding - in practice, this would use the inverse of the encoding function
    # For now, using a simple mapping
    angle = encoded_angle * math.pi / omega
    return angle


def extract_inner_edges(gray, mask=None, canny_low=50, canny_high=150):
    """
    Extract only inner edges of hoops by analyzing gradient directions
    For a hoop, inner edges have gradients pointing outward (away from center)
    
    Args:
        gray: Grayscale image
        mask: Optional binary mask
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold
    
    Returns:
        Binary edge image with only inner edges
    """
    # Apply mask if provided
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Compute gradients
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Get gradient magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)
    
    # Apply Canny to get all edges
    edges_all = cv2.Canny(blurred, canny_low, canny_high)
    
    # For inner edges, we want edges where the gradient points outward
    # This is a simplified heuristic: edges that are part of darker regions
    # For a red hoop, the inner edge is typically where we transition from red to darker center
    
    # Create a mask for inner edges by looking at local intensity
    # Inner edges should have darker regions inside (toward center)
    h, w = gray.shape
    inner_edges = np.zeros_like(edges_all)
    
    # For each edge pixel, check if the region inside (in gradient direction) is darker
    edge_pixels = np.where(edges_all > 0)
    for y, x in zip(edge_pixels[0], edge_pixels[1]):
        if 0 < y < h-1 and 0 < x < w-1:
            # Get gradient direction at this point
            grad_dir = direction[y, x]
            # Sample point slightly inward (opposite to gradient direction)
            dx = -np.cos(grad_dir) * 3
            dy = -np.sin(grad_dir) * 3
            x_inner = int(x + dx)
            y_inner = int(y + dy)
            
            # Check if inner point is valid and darker
            if 0 <= x_inner < w and 0 <= y_inner < h:
                edge_val = int(gray[y, x])
                inner_val = int(gray[y_inner, x_inner])
                if inner_val < edge_val - 10:  # Inner is darker
                    inner_edges[y, x] = 255
    
    # If we didn't get enough inner edges, use all edges but prefer darker regions
    if np.sum(inner_edges > 0) < 50:
        # Alternative: use morphological operations to find inner contours
        # Find contours and keep only those that are likely inner edges
        contours, _ = cv2.findContours(edges_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 10:  # Minimum points for a meaningful contour
                # Check if contour is likely an inner edge (has darker interior)
                mask_contour = np.zeros_like(gray)
                cv2.drawContours(mask_contour, [contour], -1, 255, 1)
                # Sample interior points
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Check if center is darker (likely inner edge)
                    if 0 <= cx < w and 0 <= cy < h:
                        # Sample around center
                        center_intensity = gray[cy, cx]
                        edge_intensity = np.mean([gray[y, x] for y, x in contour[:, 0, :] if 0 <= y < h and 0 <= x < w])
                        if center_intensity < edge_intensity - 15:  # Center is darker
                            cv2.drawContours(inner_edges, [contour], -1, 255, 1)
    
    return inner_edges


def fit_ellipse_to_partial_contour(contour, min_points=5):
    """
    Fit an ellipse to a partial contour (works for partial circles)
    
    Args:
        contour: Contour points
        min_points: Minimum number of points required
    
    Returns:
        Ellipse parameters (center, axes, angle) or None
    """
    if len(contour) < min_points:
        return None
    
    try:
        # Fit ellipse to contour
        ellipse = cv2.fitEllipse(contour.astype(np.float32))
        center = ellipse[0]
        axes = ellipse[1]
        angle = ellipse[2]
        
        # Validate ellipse
        a, b = axes[0] / 2, axes[1] / 2
        if a < 5 or b < 5:  # Too small
            return None
        
        # Check aspect ratio (should be roughly circular - stricter for hoops)
        ratio = max(a, b) / (min(a, b) + 1e-6)
        if ratio > 1.3:  # Even stricter: hoops should be very nearly circular (was 1.5)
            return None
        
        # Check angle - hoops should be roughly horizontal/vertical, not diagonal
        # Normalize angle to 0-90 range
        normalized_angle = angle % 90
        if normalized_angle > 45:
            normalized_angle = 90 - normalized_angle
        
        # Reject ellipses that are too diagonal (angle close to 45°)
        # Allow angles close to 0° or 90° (horizontal/vertical hoops)
        # But if aspect ratio is high AND angle is diagonal, definitely reject
        if ratio > 1.2 and 25 < normalized_angle < 65:  # Stricter: reject if both conditions
            return None
        elif 35 < normalized_angle < 55:  # Reject clearly diagonal ellipses regardless of ratio
            return None
        
        # Check circularity - how well the contour matches a perfect circle
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            # Circularity should be close to 1.0 for a perfect circle
            # Accept circularity > 0.5 for partial circles
            # But if aspect ratio is high, require better circularity
            min_circularity = 0.5 if ratio < 1.15 else 0.6
            if circularity < min_circularity:  # Too non-circular
                return None
        
        return {
            'center': (int(center[0]), int(center[1])),
            'axes': (int(axes[0]), int(axes[1])),
            'angle': angle,
            'xc': float(center[0]),
            'yc': float(center[1]),
            'a': float(a),
            'b': float(b),
            'orientation': math.radians(angle),
            'confidence': 1.0
        }
    except:
        return None


def hough_circles_fallback(img, mask=None, dp=1, minDist=30, param1=50, param2=30, 
                          minRadius=10, maxRadius=100, use_inner_edges=True, 
                          use_partial_circles=True):
    """
    Fallback method using contour-based ellipse fitting for partial circles
    Focuses on inner edges only
    
    Args:
        img: Input BGR image
        mask: Optional binary mask
        dp: Inverse ratio of accumulator resolution (for HoughCircles fallback)
        minDist: Minimum distance between circle centers
        param1: Upper threshold for edge detection
        param2: Accumulator threshold for center detection
        minRadius: Minimum circle radius
        maxRadius: Maximum circle radius (will be made adaptive if None)
        use_inner_edges: Whether to focus on inner edges only
        use_partial_circles: Whether to use contour fitting for partial circles
    
    Returns:
        List of detected ellipses
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    h, w = gray.shape
    
    # Make maxRadius adaptive based on image size for close hoops
    # Allow hoops up to 50% of the smaller image dimension (increased from 40%)
    if maxRadius is None or maxRadius < min(w, h) * 0.2:
        maxRadius = max(maxRadius if maxRadius else 100, int(min(w, h) * 0.5))
    
    detected_ellipses = []
    
    if use_partial_circles:
        # Method 1: Contour-based ellipse fitting (works for partial circles)
        # Try multiple edge detection strategies for better detection
        edge_images = []
        masks_to_try = [mask] if mask is not None else [None]
        
        # If we have a mask and no detections, also try without mask for close hoops
        if mask is not None:
            masks_to_try.append(None)
        
        for current_mask in masks_to_try:
            if use_inner_edges:
                # Extract only inner edges
                edges = extract_inner_edges(gray, current_mask, canny_low=param1, canny_high=param2)
                edge_images.append((edges, current_mask))
                
                # Also try with adjusted thresholds for close hoops (larger hoops need different thresholds)
                edges_adj = extract_inner_edges(gray, current_mask, canny_low=max(30, param1-20), canny_high=min(200, param2+50))
                edge_images.append((edges_adj, current_mask))
            else:
                # Use all edges
                blurred = cv2.GaussianBlur(gray, (9, 9), 2)
                if current_mask is not None:
                    blurred = cv2.bitwise_and(blurred, blurred, mask=current_mask)
                edges = cv2.Canny(blurred, param1, param2)
                edge_images.append((edges, current_mask))
                
                # Also try with adjusted thresholds
                edges_adj = cv2.Canny(blurred, max(30, param1-20), min(200, param2+50))
                edge_images.append((edges_adj, current_mask))
            
            # Also try without inner edges for close hoops (all edges)
            if use_inner_edges:
                blurred = cv2.GaussianBlur(gray, (9, 9), 2)
                if current_mask is not None:
                    blurred = cv2.bitwise_and(blurred, blurred, mask=current_mask)
                edges_all = cv2.Canny(blurred, param1, param2)
                edge_images.append((edges_all, current_mask))
                edges_all_adj = cv2.Canny(blurred, max(30, param1-20), min(200, param2+50))
                edge_images.append((edges_all_adj, current_mask))
        
        # Try all edge images
        for edges, edge_mask in edge_images:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) >= 5:  # Minimum points for ellipse fitting
                    # Fit ellipse to contour
                    ellipse = fit_ellipse_to_partial_contour(contour)
                    if ellipse is not None:
                        # Additional validation: check if it's roughly circular
                        a, b = ellipse['a'], ellipse['b']
                        ratio = max(a, b) / (min(a, b) + 1e-6)
                        if ratio < 1.3:  # Stricter: acceptable aspect ratio (was 1.5)
                            # Check size constraints - use average radius for close hoops
                            avg_radius = (a + b) / 2
                            min_radius = min(a, b)
                            max_radius = max(a, b)
                            # More flexible: accept if any radius measure is in range, or if it's a large close hoop
                            if ((minRadius <= avg_radius <= maxRadius) or 
                                (minRadius <= min_radius <= maxRadius) or
                                (minRadius <= max_radius <= maxRadius) or
                                (avg_radius > maxRadius * 0.7 and avg_radius <= maxRadius * 1.5)):  # Allow slightly larger for close hoops
                                detected_ellipses.append(ellipse)
        
        # Remove duplicates (ellipses that are very close to each other)
        if len(detected_ellipses) > 1:
            unique_ellipses = []
            for ellipse in detected_ellipses:
                is_duplicate = False
                for existing in unique_ellipses:
                    # Check if centers are very close
                    dist = math.sqrt((ellipse['xc'] - existing['xc'])**2 + (ellipse['yc'] - existing['yc'])**2)
                    if dist < minDist:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_ellipses.append(ellipse)
            detected_ellipses = unique_ellipses
        
        # If no ellipses found with partial method, try HoughCircles as backup
        # Try with and without mask for close hoops
        if len(detected_ellipses) == 0:
            masks_for_hough = [mask] if mask is not None else [None]
            if mask is not None:
                masks_for_hough.append(None)  # Also try without mask
            
            for hough_mask in masks_for_hough:
                blurred = cv2.GaussianBlur(gray, (9, 9), 2)
                if hough_mask is not None:
                    blurred = cv2.bitwise_and(blurred, blurred, mask=hough_mask)
                
                # Try multiple parameter sets for different hoop sizes
                param_sets = [
                    (param1, param2),  # Original
                    (max(30, param1-20), min(200, param2+50)),  # Adjusted for larger hoops
                    (max(20, param1-30), min(250, param2+100)),  # More adjusted for very large hoops
                ]
                
                for p1, p2 in param_sets:
                    circles = cv2.HoughCircles(
                        blurred,
                        cv2.HOUGH_GRADIENT,
                        dp=dp,
                        minDist=minDist,
                        param1=p1,
                        param2=p2,
                        minRadius=minRadius,
                        maxRadius=maxRadius
                    )
                    
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        for (x, y, r) in circles:
                            detected_ellipses.append({
                                'center': (int(x), int(y)),
                                'axes': (int(r * 2), int(r * 2)),
                                'angle': 0.0,
                                'xc': float(x),
                                'yc': float(y),
                                'a': float(r),
                                'b': float(r),
                                'orientation': 0.0,
                                'confidence': 1.0
                            })
                        break  # Found circles, no need to try other param sets
                
                if len(detected_ellipses) > 0:
                    break  # Found ellipses, no need to try other masks
    else:
        # Original HoughCircles method
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        if mask is not None:
            blurred = cv2.bitwise_and(blurred, blurred, mask=mask)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=minDist,
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                detected_ellipses.append({
                    'center': (int(x), int(y)),
                    'axes': (int(r * 2), int(r * 2)),
                    'angle': 0.0,
                    'xc': float(x),
                    'yc': float(y),
                    'a': float(r),
                    'b': float(r),
                    'orientation': 0.0,
                    'confidence': 1.0
                })
    
    return detected_ellipses


def load_ellipse_yolo_model(model_path=None):
    """
    Load Ellipse-YOLO model from checkpoint
    
    Args:
        model_path: Path to model weights file (.pt or .pth)
    
    Returns:
        Loaded model or None if not available
    """
    if not TORCH_AVAILABLE:
        return None
    
    if model_path is None:
        # Try to find model in common locations
        possible_paths = [
            Path(__file__).parent.parent / "models" / "ellipse_yolo.pt",
            Path(__file__).parent.parent.parent.parent / "models" / "ellipse_yolo.pt",
        ]
        
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
    
    if model_path is None or not Path(model_path).exists():
        return None
    
    try:
        # Load model - this would need to match the actual model architecture
        # For now, returning None as placeholder
        # model = torch.load(model_path, map_location='cpu')
        # model.eval()
        # return model
        return None
    except Exception as e:
        print(f"Warning: Could not load model from {model_path}: {e}")
        return None


def detect_hoops_ellipse_yolo(img,
                               # Model parameters
                               model_path=None,
                               use_model=True,
                               # HSV masking parameters
                               use_red_mask=True,
                               lower_red1=(0, 50, 50), upper_red1=(10, 255, 255),
                               lower_red2=(170, 50, 50), upper_red2=(180, 255, 255),
                               # Fallback HoughCircles parameters (if model not available)
                               fallback_dp=1,
                               fallback_minDist=30,
                               fallback_param1=50,
                               fallback_param2=30,
                               fallback_minRadius=10,
                               fallback_maxRadius=None,  # None = adaptive based on image size
                               use_inner_edges=True,  # Focus on inner edges only
                               use_partial_circles=True,  # Support partial circles
                               # Selection parameters
                               reference_point=None,
                               draw_all_ellipses=False,
                               # Model inference parameters
                               conf_threshold=0.25,
                               iou_threshold=0.45,
                               img_size=640):
    """
    Detect hoops using Ellipse-YOLO model (or fallback method)
    Supports partial circles and focuses on inner edges only for tighter contours
    
    Args:
        img: Input BGR image
        model_path: Path to trained model weights
        use_model: Whether to try using the model (falls back if not available)
        use_red_mask: Whether to use red color masking
        lower_red1, upper_red1: First red range in HSV
        lower_red2, upper_red2: Second red range in HSV (wrap-around)
        fallback_*: Parameters for HoughCircles fallback method
        use_inner_edges: If True, only detect inner edges of hoops (default: True)
        use_partial_circles: If True, use contour fitting for partial circles (default: True)
        reference_point: (x, y) reference point for selection (default: image center)
        draw_all_ellipses: If True, draw all detected ellipses; if False, only nearest
        conf_threshold: Confidence threshold for model predictions
        iou_threshold: IoU threshold for NMS
        img_size: Input image size for model (should match training)
    
    Returns:
        (result_image, detected_ellipses, nearest_ellipse_idx, edge_image, mask)
    """
    h, w = img.shape[:2]
    
    # Step 1: Create red mask if enabled
    mask = None
    if use_red_mask:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = make_red_mask(hsv, lower_red1, upper_red1, lower_red2, upper_red2)
    
    # Step 2: Try to use model if available
    detected_ellipses = []
    model = None
    
    if use_model and TORCH_AVAILABLE:
        model = load_ellipse_yolo_model(model_path)
    
    if model is not None:
        # Use Ellipse-YOLO model for detection
        try:
            # Preprocess image
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (img_size, img_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                predictions = model(img_tensor)
            
            # Post-process predictions
            # This would need to match the actual model output format
            # For now, placeholder - would decode rotated bounding boxes
            # and extract ellipse parameters (center, axes, angle)
            
            # Placeholder: would process model output here
            pass
            
        except Exception as e:
            print(f"Warning: Model inference failed: {e}")
            # Fall through to fallback method
    
    # Step 3: Use fallback method if model not available or failed
    if len(detected_ellipses) == 0:
        detected_ellipses = hough_circles_fallback(
            img,
            mask=mask,
            dp=fallback_dp,
            minDist=fallback_minDist,
            param1=fallback_param1,
            param2=fallback_param2,
            minRadius=fallback_minRadius,
            maxRadius=fallback_maxRadius,
            use_inner_edges=use_inner_edges,
            use_partial_circles=use_partial_circles
        )
    
    # Step 4: Select nearest ellipse if multiple detected
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
    
    # Step 5: Draw results
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
    
    # Create edge image for visualization (using inner edges if enabled)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if use_inner_edges:
        # Use inner edges only for visualization
        edge_image = extract_inner_edges(gray, mask, canny_low=fallback_param1, canny_high=fallback_param2)
    else:
        # Use all edges
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        edge_image = cv2.Canny(blurred, fallback_param1, fallback_param2)
    
    return result_img, detected_ellipses, nearest_idx, edge_image, mask


def process_video(video_path, output_dir, canny_edges_dir, output_video_path=None, **kwargs):
    """
    Process video file with Ellipse-YOLO pipeline
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        output_video_path: Path to save output video (optional)
        **kwargs: Additional parameters for detect_hoops_ellipse_yolo
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 fps if not available
    print(f"  Video has {total_frames} frames at {fps:.2f} fps")
    
    # Set up video writer if output path is provided
    video_writer = None
    if output_video_path:
        # Get frame dimensions from first frame
        ret, first_frame = cap.read()
        if ret:
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))
            if not video_writer.isOpened():
                print(f"Warning: Could not create video file {output_video_path}")
                video_writer = None
            else:
                print(f"  Creating output video: {output_video_path}")
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    processed = 0
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hoops
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ellipse_yolo(
            frame, **kwargs
        )
        
        # Save result image
        output_file = output_path / f"frame_{frame_num:06d}_result.jpg"
        cv2.imwrite(str(output_file), result_img)
        
        # Save Canny edge image
        canny_file = canny_path / f"frame_{frame_num:06d}_edges.jpg"
        cv2.imwrite(str(canny_file), edge_img)
        
        # Write to video if writer is available
        if video_writer is not None:
            video_writer.write(result_img)
        
        processed += 1
        frame_num += 1
        
        if processed % 10 == 0:
            print(f"  Processed {processed}/{total_frames} frames...")
            if len(ellipses) > 0:
                print(f"    Detected {len(ellipses)} ellipse(s), nearest: {nearest_idx}")
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"\n✓ Video saved to: {output_video_path}")
    
    print(f"\n✓ Processed {processed} frames")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def process_frames(input_dir, output_dir, canny_edges_dir, num_frames=None, **kwargs):
    """
    Process frames with Ellipse-YOLO pipeline
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        num_frames: Number of frames to process (None = process all available)
        **kwargs: Additional parameters for detect_hoops_ellipse_yolo
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
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ellipse_yolo(
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
    Process test images with Ellipse-YOLO pipeline
    
    Args:
        test_images_dir: Directory containing test images
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        **kwargs: Additional parameters for detect_hoops_ellipse_yolo
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
    
    # Check if model is available
    if TORCH_AVAILABLE:
        print("PyTorch is available. Model loading will be attempted if model_path is provided.")
    else:
        print("PyTorch not available. Using fallback HoughCircles method.")
    
    processed = 0
    for img_file in sorted(image_files):
        print(f"\nProcessing: {img_file.name}")
        
        # Load image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  Could not load {img_file.name}, skipping...")
            continue
        
        # Detect hoops
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ellipse_yolo(
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
    """Process test images with Ellipse-YOLO pipeline"""
    test_images_dir = Path(__file__).parent.parent.parent.parent / "input" / "test_images"
    output_dir = Path(__file__).parent.parent / "output" / "test_results"
    canny_edges_dir = Path(__file__).parent.parent / "output" / "test_canny_edges"
    
    print(f"Processing test images with Ellipse-YOLO pipeline...")
    print(f"Input: {test_images_dir}")
    print(f"Results: {output_dir}")
    print(f"Canny edges: {canny_edges_dir}\n")
    
    process_test_images(
        test_images_dir,
        output_dir,
        canny_edges_dir,
        use_red_mask=True,  # Use red masking
        use_model=True,  # Try to use model if available
        model_path=None,  # Set to model path if you have trained model
        use_inner_edges=True,  # Focus on inner edges only
        use_partial_circles=True,  # Support partial circles
        draw_all_ellipses=False,  # Only draw nearest ellipse
    )


if __name__ == "__main__":
    main()

