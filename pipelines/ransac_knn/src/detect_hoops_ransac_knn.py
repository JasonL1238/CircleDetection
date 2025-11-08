"""
RANSAC + k-NN Hoop Detection Pipeline
Combines edge detection, clustering, RANSAC ellipse fitting, and k-NN selection
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
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


def extract_edge_points(img, mask=None, canny_low=50, canny_high=150, use_gradient_filter=True):
    """
    Extract edge points from the image using Canny edge detection with optional gradient filtering
    
    Args:
        img: Input image (BGR or grayscale)
        mask: Optional binary mask to restrict edge detection to specific regions
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        use_gradient_filter: Whether to filter edges by gradient direction
    
    Returns:
        Array of (x, y) edge point coordinates, edge image
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply mask if provided
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Optional: gradient direction filtering to suppress background noise
    if use_gradient_filter and mask is not None:
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        # Only apply where gradient is significant (if there are non-zero gradients)
        non_zero_gradients = gradient_magnitude[gradient_magnitude > 0]
        if len(non_zero_gradients) > 0:
            threshold = np.percentile(non_zero_gradients, 25)
            edges[gradient_magnitude < threshold] = 0
    
    # Optional: enforce mask again
    if mask is not None:
        edges[mask == 0] = 0
    
    # Extract edge point coordinates
    edge_points = np.column_stack(np.where(edges > 0))
    
    # Convert from (row, col) to (x, y)
    if len(edge_points) > 0:
        edge_points = edge_points[:, [1, 0]]  # Swap columns
    
    return edge_points, edges


def find_inner_outer_contours(mask):
    """
    Find inner and outer contour pairs using hierarchy
    
    Args:
        mask: Binary mask
    
    Returns:
        List of (outer_contour, inner_contour) pairs, or empty list
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None or len(contours) == 0:
        return []
    
    pairs = []
    hierarchy = hierarchy[0]  # Get first (and only) hierarchy array
    
    for i, h in enumerate(hierarchy):
        parent = h[3]  # Parent contour index
        if parent != -1:  # Has a parent (inner contour)
            outer_contour = contours[parent]
            inner_contour = contours[i]
            # Check if both are substantial
            if cv2.contourArea(outer_contour) > 50 and cv2.contourArea(inner_contour) > 20:
                pairs.append((outer_contour, inner_contour))
    
    return pairs


def refine_center_with_moments(mask_region):
    """
    Refine center using moments of the mask region
    
    Args:
        mask_region: Binary mask region
    
    Returns:
        (cx, cy) refined center, or None if invalid
    """
    M = cv2.moments(mask_region)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy)


def fit_ellipse_ransac(points, min_samples=5, max_iterations=1000, 
                       distance_threshold=5.0, min_inliers=10,
                       img_height=None, img_width=None, use_ransac=True):
    """
    Fit an ellipse to points using RANSAC or direct fitting
    
    Args:
        points: Array of (x, y) coordinates
        min_samples: Minimum number of points to fit ellipse
        max_iterations: Maximum RANSAC iterations
        distance_threshold: Distance threshold for inliers
        min_inliers: Minimum number of inliers to accept ellipse
        img_height: Image height for sanity checks
        img_width: Image width for sanity checks
        use_ransac: If False and clusters are clean, use direct fitting
    
    Returns:
        (center_x, center_y, major_axis, minor_axis, angle, inlier_count) or None
    """
    if len(points) < min_samples:
        return None
    
    # If clusters are clean, use direct fitting instead of RANSAC
    if not use_ransac and len(points) >= 5:
        try:
            ellipse = cv2.fitEllipse(points.astype(np.float32))
            center = ellipse[0]
            axes = ellipse[1]
            angle = ellipse[2]
            center_x, center_y = center
            a, b = axes[0] / 2, axes[1] / 2  # Semi-axes
            
            # Sanity checks with stricter shape prior
            if img_height is not None and img_width is not None:
                # Too big
                if a > img_width or b > img_height:
                    return None
                
                # Aspect ratio - enforce shape prior (eccentricity < 2.5)
                ratio = max(a, b) / (min(a, b) + 1e-6)
                if ratio > 2.5:  # Stricter than before (was 3.0)
                    return None
                
                # Center way off
                if not (0 <= center_x < img_width and 0 <= center_y < img_height):
                    return None
            
            return (center_x, center_y, axes[0], axes[1], angle, len(points))
        except:
            return None
    
    # Use RANSAC
    best_ellipse = None
    best_inliers = []
    best_inlier_count = 0
    
    np.random.seed(42)  # For reproducibility
    
    for _ in range(max_iterations):
        # Randomly sample 5 points (minimum for ellipse fitting)
        if len(points) < 5:
            break
        
        sample_indices = np.random.choice(len(points), size=min(5, len(points)), replace=False)
        sample_points = points[sample_indices]
        
        try:
            # Fit ellipse to sample points
            # OpenCV requires at least 5 points
            if len(sample_points) >= 5:
                ellipse = cv2.fitEllipse(sample_points.astype(np.float32))
                
                # Extract ellipse parameters
                center = ellipse[0]
                axes = ellipse[1]
                angle = ellipse[2]
                
                center_x, center_y = center
                a, b = axes[0] / 2, axes[1] / 2  # Semi-axes
                
                # Sanity checks with stricter shape prior
                if img_height is not None and img_width is not None:
                    # Too big
                    if a > img_width or b > img_height:
                        continue
                    
                    # Aspect ratio - enforce shape prior (eccentricity < 2.5)
                    ratio = max(a, b) / (min(a, b) + 1e-6)
                    if ratio > 2.5:  # Stricter than before (was 3.0)
                        continue
                    
                    # Center way off
                    if not (0 <= center_x < img_width and 0 <= center_y < img_height):
                        continue
                
                # Calculate distances from all points to ellipse
                cos_a = math.cos(math.radians(angle))
                sin_a = math.sin(math.radians(angle))
                
                inliers = []
                for point in points:
                    x, y = point
                    # Translate to center
                    dx = x - center_x
                    dy = y - center_y
                    
                    # Rotate to align with ellipse axes
                    rx = dx * cos_a + dy * sin_a
                    ry = -dx * sin_a + dy * cos_a
                    
                    # Calculate distance to ellipse (normalized)
                    dist = (rx**2 / (a**2 + 1e-6)) + (ry**2 / (b**2 + 1e-6))
                    dist = abs(math.sqrt(dist) - 1.0) * min(a, b)
                    
                    if dist < distance_threshold:
                        inliers.append(point)
                
                inlier_count = len(inliers)
                
                # Update best ellipse if this one has more inliers
                if inlier_count > best_inlier_count and inlier_count >= min_inliers:
                    best_inlier_count = inlier_count
                    best_inliers = inliers
                    best_ellipse = (center_x, center_y, axes[0], axes[1], angle, inlier_count)
        
        except:
            continue
    
    return best_ellipse


def cluster_edge_points(points, eps=50, min_samples=10):
    """
    Cluster edge points using DBSCAN to separate different hoops
    
    Args:
        points: Array of (x, y) coordinates
        eps: Maximum distance between points in same cluster
        min_samples: Minimum points to form a cluster
    
    Returns:
        List of point arrays, one per cluster
    """
    if len(points) == 0:
        return []
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Group points by cluster
    clusters = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = points[labels == label]
        clusters.append(cluster_points)
    
    return clusters


def select_nearest_hoop(ellipses, reference_point=None, img_height=None,
                        distance_weight=0.5, area_weight=0.3, y_weight=0.2):
    """
    Use improved k-NN to select the nearest hoop from detected ellipses
    Includes distance, area, and vertical position (y-coordinate)
    
    Args:
        ellipses: List of (center_x, center_y, major_axis, minor_axis, angle, inlier_count)
        reference_point: (x, y) reference point (default: image center)
        img_height: Image height for y-position normalization
        distance_weight: Weight for distance feature (default: 0.5)
        area_weight: Weight for area feature (default: 0.3)
        y_weight: Weight for y-position (default: 0.2)
    
    Returns:
        Index of nearest ellipse, or None if no ellipses
    """
    if len(ellipses) == 0:
        return None
    
    if len(ellipses) == 1:
        return 0
    
    # Extract features: [center_x, center_y, area, major_axis, y_position]
    features = []
    for ellipse in ellipses:
        cx, cy, major, minor, angle, inliers = ellipse
        area = math.pi * (major / 2) * (minor / 2)
        # Normalize y by image height if available
        y_norm = cy / img_height if img_height else cy
        features.append([cx, cy, area, major, y_norm])
    
    features = np.array(features)
    
    # Normalize features
    features_norm = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        if col.max() > col.min():
            features_norm[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-6)
        else:
            features_norm[:, i] = 0.5  # Neutral value if all same
    features = features_norm
    
    # Use reference point (default: image center or mean)
    if reference_point is None:
        # Use mean center as reference
        ref_x = np.mean(features[:, 0])
        ref_y = np.mean(features[:, 1])
        ref_area = np.mean(features[:, 2])
        ref_y_pos = np.mean(features[:, 4])
    else:
        ref_x, ref_y = reference_point
        ref_area = np.mean(features[:, 2])
        ref_y_pos = ref_y / img_height if img_height else ref_y
    
    # Normalize reference
    ref_features = features.copy()
    for i in range(features.shape[1]):
        col = features[:, i]
        if col.max() > col.min():
            ref_features[0, i] = (ref_features[0, i] - col.min()) / (col.max() - col.min() + 1e-6)
    ref_x_norm = ref_features[0, 0]
    ref_y_norm = ref_features[0, 1]
    ref_area_norm = ref_features[0, 2]
    ref_y_pos_norm = ref_features[0, 4] if features.shape[1] > 4 else 0.5
    
    # Calculate weighted scores (lower is better)
    scores = []
    for i, feat in enumerate(features):
        # Distance component (normalized)
        dist_xy = math.sqrt((feat[0] - ref_x_norm)**2 + (feat[1] - ref_y_norm)**2)
        
        # Area component (larger area = closer, so invert)
        dist_area = 1.0 - feat[2]  # Invert so larger area = smaller distance
        
        # Y-position component (closer to reference y = better)
        dist_y = abs(feat[4] - ref_y_pos_norm) if features.shape[1] > 4 else 0
        
        # Combined score
        score = (distance_weight * dist_xy + 
                area_weight * dist_area + 
                y_weight * dist_y)
        
        scores.append(score)
    
    # Return index of nearest (lowest score)
    nearest_idx = np.argmin(scores)
    return nearest_idx


def detect_hoops_ransac_knn(img, 
                            # HSV masking parameters
                            use_red_mask=True,
                            lower_red1=(0, 50, 50), upper_red1=(10, 255, 255),
                            lower_red2=(170, 50, 50), upper_red2=(180, 255, 255),
                            # Edge detection parameters
                            canny_low=50, canny_high=150,
                            # Clustering parameters
                            use_clustering=True, cluster_eps=15, cluster_min_samples=30,
                            # RANSAC parameters
                            ransac_min_samples=5, ransac_max_iterations=1000,
                            ransac_distance_threshold=5.0, ransac_min_inliers=10,
                            use_ransac=False,  # If clusters are clean, skip RANSAC
                            # k-NN parameters
                            reference_point=None, use_area=True,
                            area_weight=0.3, distance_weight=0.7,
                            # Visualization
                            draw_all_ellipses=False):
    """
    Complete pipeline: Red masking -> Edge detection -> Clustering -> RANSAC/Direct fitting -> k-NN
    
    Args:
        img: Input BGR image
        use_red_mask: Whether to use red color masking
        lower_red1, upper_red1: First red range in HSV
        lower_red2, upper_red2: Second red range in HSV (wrap-around)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        use_clustering: Enable DBSCAN clustering
        cluster_eps: Maximum distance between points in same cluster (tighter: 15)
        cluster_min_samples: Minimum points to form a cluster (tighter: 30)
        ransac_min_samples: Minimum points to fit ellipse
        ransac_max_iterations: Maximum RANSAC iterations
        ransac_distance_threshold: Distance threshold for inliers
        ransac_min_inliers: Minimum inliers to accept ellipse
        use_ransac: If False and clusters are clean, use direct fitting
        reference_point: (x, y) reference point (default: image center)
        use_area: Consider area in selection
        area_weight: Weight for area feature
        distance_weight: Weight for distance feature
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
    
    # Step 2: Extract edge points (only on red regions if mask provided)
    edge_points, edge_image = extract_edge_points(img, mask, canny_low, canny_high)
    
    if len(edge_points) == 0:
        return img.copy(), [], None, edge_image, mask
    
    # Step 3: Cluster edge points (optional)
    if use_clustering:
        clusters = cluster_edge_points(edge_points, cluster_eps, cluster_min_samples)
        if len(clusters) == 0:
            # If clustering found nothing, use all points as one cluster
            clusters = [edge_points]
    else:
        clusters = [edge_points]
    
    # Step 4: Apply RANSAC or direct ellipse fitting per cluster
    detected_ellipses = []
    for cluster in clusters:
        ellipse = fit_ellipse_ransac(
            cluster, 
            ransac_min_samples, 
            ransac_max_iterations,
            ransac_distance_threshold,
            ransac_min_inliers,
            img_height=h,
            img_width=w,
            use_ransac=use_ransac
        )
        if ellipse is not None:
            detected_ellipses.append(ellipse)
    
    # Step 4.5: Refine centers using moments if mask available
    if mask is not None and len(detected_ellipses) > 0:
        # Try to find inner/outer contour pairs
        contour_pairs = find_inner_outer_contours(mask)
        
        # Refine ellipse centers using moments or inner/outer averaging
        refined_ellipses = []
        for ellipse in detected_ellipses:
            cx, cy, major, minor, angle, inliers = ellipse
            
            # Try to find matching contour pair
            refined_center = None
            for outer_contour, inner_contour in contour_pairs:
                # Check if ellipse center is near this pair
                outer_cx, outer_cy = np.mean(outer_contour[:, 0, 0]), np.mean(outer_contour[:, 0, 1])
                inner_cx, inner_cy = np.mean(inner_contour[:, 0, 0]), np.mean(inner_contour[:, 0, 1])
                
                # If ellipse center is close to average of inner/outer
                avg_cx = (outer_cx + inner_cx) / 2
                avg_cy = (outer_cy + inner_cy) / 2
                
                dist_to_avg = math.sqrt((cx - avg_cx)**2 + (cy - avg_cy)**2)
                if dist_to_avg < max(major, minor) / 2:
                    # Use averaged center
                    refined_center = (avg_cx, avg_cy)
                    break
            
            # If no pair found, try moments on mask region around ellipse
            if refined_center is None:
                # Create a small mask region around the ellipse
                mask_region = np.zeros_like(mask)
                cv2.ellipse(mask_region, (int(cx), int(cy)), 
                           (int(major/2), int(minor/2)), angle, 0, 360, 255, -1)
                mask_region = cv2.bitwise_and(mask_region, mask)
                
                moment_center = refine_center_with_moments(mask_region)
                if moment_center is not None:
                    refined_center = moment_center
                else:
                    refined_center = (cx, cy)
            
            # Use refined center
            refined_ellipses.append((refined_center[0], refined_center[1], major, minor, angle, inliers))
        
        detected_ellipses = refined_ellipses
    
    # Step 5: Use improved k-NN to select nearest hoop
    nearest_idx = None
    if len(detected_ellipses) > 0:
        # Get image center as default reference
        if reference_point is None:
            reference_point = (w // 2, h // 2)
        
        nearest_idx = select_nearest_hoop(
            detected_ellipses,
            reference_point,
            img_height=h,
            distance_weight=distance_weight,
            area_weight=area_weight,
            y_weight=0.2  # Add y-position weight
        )
    
    # Step 6: Draw results
    result_img = img.copy()
    
    if draw_all_ellipses:
        # Draw all detected ellipses in blue
        for i, ellipse in enumerate(detected_ellipses):
            cx, cy, major, minor, angle, inliers = ellipse
            center = (int(cx), int(cy))
            axes = (int(major / 2), int(minor / 2))
            cv2.ellipse(result_img, center, axes, angle, 0, 360, (255, 0, 0), 2)
            cv2.circle(result_img, center, 5, (255, 0, 0), -1)
    else:
        # Draw only nearest ellipse in green
        if nearest_idx is not None:
            ellipse = detected_ellipses[nearest_idx]
            cx, cy, major, minor, angle, inliers = ellipse
            center = (int(cx), int(cy))
            axes = (int(major / 2), int(minor / 2))
            cv2.ellipse(result_img, center, axes, angle, 0, 360, (0, 255, 0), 3)
            cv2.circle(result_img, center, 5, (0, 255, 0), -1)
            
            # Draw big blue dot at determined center
            cv2.circle(result_img, center, 20, (255, 0, 0), -1)  # Big blue filled circle
            
            # Draw reference point
            if reference_point:
                cv2.circle(result_img, reference_point, 5, (0, 0, 255), -1)
    
    return result_img, detected_ellipses, nearest_idx, edge_image, mask


def process_video(video_path, output_dir, canny_edges_dir, **kwargs):
    """
    Process video file with RANSAC + k-NN pipeline
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        **kwargs: Additional parameters for detect_hoops_ransac_knn
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
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ransac_knn(
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
    Process frames with RANSAC + k-NN pipeline
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        num_frames: Number of frames to process (None = process all available)
        **kwargs: Additional parameters for detect_hoops_ransac_knn
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
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ransac_knn(
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
    Process test images with RANSAC + k-NN pipeline
    
    Args:
        test_images_dir: Directory containing test images
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        **kwargs: Additional parameters for detect_hoops_ransac_knn
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
        result_img, ellipses, nearest_idx, edge_img, mask = detect_hoops_ransac_knn(
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
                cx, cy, major, minor, angle, inliers = ellipse
                print(f"    Selected ellipse: center=({cx:.1f}, {cy:.1f}), "
                      f"axes=({major:.1f}, {minor:.1f}), inliers={inliers}")
        else:
            print(f"  ✗ No ellipses detected")
        
        processed += 1
    
    print(f"\n✓ Processed {processed} test images")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def main():
    """Process test images with RANSAC + k-NN pipeline"""
    test_images_dir = Path(__file__).parent.parent.parent.parent / "input" / "test_images"
    output_dir = Path(__file__).parent.parent / "output" / "test_results"
    canny_edges_dir = Path(__file__).parent.parent / "output" / "test_canny_edges"
    
    print(f"Processing test images with RANSAC + k-NN pipeline...")
    print(f"Input: {test_images_dir}")
    print(f"Results: {output_dir}")
    print(f"Canny edges: {canny_edges_dir}\n")
    
    process_test_images(
        test_images_dir,
        output_dir, 
        canny_edges_dir,
        use_red_mask=True,  # Use red masking
        use_clustering=True,
        cluster_eps=15,  # Tighter clustering
        cluster_min_samples=30,  # More points required
        use_ransac=False,  # Skip RANSAC if clusters are clean
        draw_all_ellipses=False,  # Only draw nearest ellipse
    )


if __name__ == "__main__":
    main()

