"""
Partial Ellipse Fit Hoop Detection Pipeline
Robust detection for partially visible hoops and close-range scenarios using direct least-squares fitting
"""

import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
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


def fit_ellipse_direct(points):
    """
    Fit ellipse to points using Fitzgibbon's Direct Least Squares method
    Works even for partial arcs (as little as 25% of the ellipse)
    
    Args:
        points: Nx2 array of (x, y) points
    
    Returns:
        Tuple of (a, b, c, d, e, f) conic coefficients, or None if fitting fails
    """
    if len(points) < 5:
        return None
    
    x = points[:, 0][:, np.newaxis]
    y = points[:, 1][:, np.newaxis]
    
    # Build design matrix D = [x^2, xy, y^2, x, y, 1]
    D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    
    # Build scatter matrix
    S = np.dot(D.T, D)
    
    # Build constraint matrix C (for ellipse: 4ac - b^2 > 0)
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    
    try:
        # Solve generalized eigenvalue problem: S * v = lambda * C * v
        # We want the eigenvector corresponding to the largest positive eigenvalue
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S).dot(C))
        
        # Find eigenvector with largest absolute eigenvalue that satisfies constraint
        valid_indices = []
        for i, val in enumerate(eig_vals):
            v = eig_vecs[:, i]
            a, b, c = v[0], v[1], v[2]
            # Check ellipse constraint: 4ac - b^2 > 0
            if 4 * a * c - b * b > 0:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return None
        
        # Use the eigenvector with largest eigenvalue
        best_idx = valid_indices[np.argmax(np.abs(eig_vals[valid_indices]))]
        eig_vec = eig_vecs[:, best_idx]
        
        return tuple(eig_vec)
    except:
        return None


def ellipse_center(a, b, c, d, e, f):
    """
    Extract center from conic coefficients
    
    Args:
        a, b, c, d, e, f: Conic coefficients (ax^2 + bxy + cy^2 + dx + ey + f = 0)
    
    Returns:
        (cx, cy) center coordinates
    """
    den = b * b - 4 * a * c
    if abs(den) < 1e-10:
        return None
    
    cx = (2 * c * d - b * e) / den
    cy = (2 * a * e - b * d) / den
    return (cx, cy)


def ellipse_parameters(a, b, c, d, e, f):
    """
    Convert conic coefficients to standard ellipse parameters
    Uses standard formula for converting conic to ellipse parameters
    
    Args:
        a, b, c, d, e, f: Conic coefficients (ax^2 + bxy + cy^2 + dx + ey + f = 0)
    
    Returns:
        Dictionary with center, axes, and angle, or None if invalid
    """
    center = ellipse_center(a, b, c, d, e, f)
    if center is None:
        return None
    
    cx, cy = center
    
    # Translate conic to center: substitute x' = x - cx, y' = y - cy
    # The constant term becomes: f' = a*cx^2 + b*cx*cy + c*cy^2 + d*cx + e*cy + f
    f_translated = a * cx * cx + b * cx * cy + c * cy * cy + d * cx + e * cy + f
    
    if abs(f_translated) < 1e-10:
        return None
    
    # Normalize so that f' = -1 (standard ellipse form: Ax^2 + Bxy + Cy^2 = 1)
    scale = -1.0 / f_translated
    
    a_norm = a * scale
    b_norm = b * scale
    c_norm = c * scale
    
    # Calculate rotation angle
    if abs(b_norm) < 1e-10:
        if a_norm < c_norm:
            angle_rad = 0
        else:
            angle_rad = math.pi / 2
    else:
        angle_rad = 0.5 * math.atan2(b_norm, a_norm - c_norm)
    
    angle = math.degrees(angle_rad)
    
    # Rotate to align with axes
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # After rotation, the conic becomes A'x'^2 + C'y'^2 = 1
    A_rotated = a_norm * cos_a * cos_a + b_norm * cos_a * sin_a + c_norm * sin_a * sin_a
    C_rotated = a_norm * sin_a * sin_a - b_norm * cos_a * sin_a + c_norm * cos_a * cos_a
    
    if A_rotated <= 0 or C_rotated <= 0:
        return None
    
    # Semi-axes: 1/sqrt(A') and 1/sqrt(C')
    semi_major = 1.0 / math.sqrt(min(A_rotated, C_rotated))
    semi_minor = 1.0 / math.sqrt(max(A_rotated, C_rotated))
    
    # Ensure major > minor
    if semi_major < semi_minor:
        semi_major, semi_minor = semi_minor, semi_major
        angle += 90
        if angle >= 180:
            angle -= 180
    
    return {
        'center': (float(cx), float(cy)),
        'axes': (float(semi_major * 2), float(semi_minor * 2)),
        'angle': float(angle),
        'xc': float(cx),
        'yc': float(cy),
        'a': float(semi_major),
        'b': float(semi_minor),
        'orientation': math.radians(angle),
        'confidence': 1.0
    }


def extract_inner_edges(gray, mask=None, canny_low=50, canny_high=150):
    """
    Extract only inner edges of hoops by analyzing gradient directions
    
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
        contours, _ = cv2.findContours(edges_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 10:  # Minimum points for a meaningful contour
                # Check if contour is likely an inner edge (has darker interior)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Check if center is darker (likely inner edge)
                    if 0 <= cx < w and 0 <= cy < h:
                        center_intensity = gray[cy, cx]
                        edge_points = contour[:, 0, :]
                        valid_points = [(y, x) for y, x in edge_points if 0 <= y < h and 0 <= x < w]
                        if len(valid_points) > 0:
                            edge_intensity = np.mean([gray[y, x] for y, x in valid_points])
                            if center_intensity < edge_intensity - 15:  # Center is darker
                                cv2.drawContours(inner_edges, [contour], -1, 255, 1)
    
    return inner_edges


def detect_hoops_partial_ellipse(img,
                                 # HSV masking parameters
                                 use_red_mask=True,
                                 lower_red1=(0, 50, 50), upper_red1=(10, 255, 255),
                                 lower_red2=(170, 50, 50), upper_red2=(180, 255, 255),
                                 # Edge detection parameters
                                 canny_low=50,
                                 canny_high=150,
                                 use_inner_edges=True,
                                 # Clustering parameters
                                 cluster_eps=20,  # Increased for larger clusters
                                 cluster_min_samples=30,  # Increased for more robust clusters
                                 # Selection parameters
                                 reference_point=None,
                                 prefer_largest=True,
                                 prefer_highest_gradient=True,
                                 # Fallback parameters
                                 use_circle_fallback=True,
                                 min_points_for_ellipse=5,
                                 # Visualization
                                 draw_all_ellipses=False):
    """
    Detect hoops using direct least-squares ellipse fitting on partial arcs
    Works for partially visible hoops and close-range scenarios
    
    Args:
        img: Input BGR image
        use_red_mask: Whether to use red color masking
        lower_red1, upper_red1: First red range in HSV
        lower_red2, upper_red2: Second red range in HSV (wrap-around)
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold
        use_inner_edges: Whether to focus on inner edges only
        cluster_eps: Maximum distance between points in same cluster (DBSCAN)
        cluster_min_samples: Minimum points to form a cluster
        reference_point: (x, y) reference point for selection (default: image center)
        prefer_largest: Prefer ellipses with larger minor axis
        prefer_highest_gradient: Prefer ellipses with higher gradient density
        use_circle_fallback: Fallback to circle fitting if ellipse fitting fails
        min_points_for_ellipse: Minimum points required for ellipse fitting
        draw_all_ellipses: If True, draw all detected ellipses; if False, only best
    
    Returns:
        (result_image, detected_ellipses, best_ellipse_idx, edge_image, mask)
    """
    h, w = img.shape[:2]
    
    # Step 1: Create red mask if enabled
    mask = None
    if use_red_mask:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = make_red_mask(hsv, lower_red1, upper_red1, lower_red2, upper_red2)
    
    # Step 2: Extract edge points
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if use_inner_edges:
        edges = extract_inner_edges(gray, mask, canny_low, canny_high)
    else:
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        if mask is not None:
            blurred = cv2.bitwise_and(blurred, blurred, mask=mask)
        edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # Extract edge point coordinates
    y_coords, x_coords = np.nonzero(edges)
    if len(x_coords) == 0:
        return img.copy(), [], None, edges, mask
    
    points = np.column_stack((x_coords, y_coords))
    
    # Step 3: Cluster edges into arc-like groups using DBSCAN
    clustering = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(points)
    labels = clustering.labels_
    
    # Get unique cluster labels (excluding noise: -1)
    unique_labels = [i for i in np.unique(labels) if i != -1]
    
    if len(unique_labels) == 0:
        return img.copy(), [], None, edges, mask
    
    # Step 3.5: Merge nearby clusters that might be part of the same hoop
    # Calculate cluster centers and merge if they're close
    cluster_centers = []
    cluster_points_list = []
    for label in unique_labels:
        cluster = points[labels == label]
        if len(cluster) >= cluster_min_samples:
            center = np.mean(cluster, axis=0)
            cluster_centers.append(center)
            cluster_points_list.append(cluster)
    
    # Merge clusters that are close together (within 2*cluster_eps)
    merged_clusters = []
    used = [False] * len(cluster_points_list)
    
    for i, (center, cluster) in enumerate(zip(cluster_centers, cluster_points_list)):
        if used[i]:
            continue
        
        merged = cluster.copy()
        used[i] = True
        
        # Find nearby clusters to merge
        for j in range(i + 1, len(cluster_points_list)):
            if used[j]:
                continue
            dist = np.linalg.norm(center - cluster_centers[j])
            if dist < cluster_eps * 3:  # Merge if within 3x eps
                merged = np.vstack([merged, cluster_points_list[j]])
                used[j] = True
        
        merged_clusters.append(merged)
    
    # Step 4: Fit ellipses to each cluster (merged or original)
    detected_ellipses = []
    ellipse_scores = []
    
    for cluster in merged_clusters:
        
        if len(cluster) < min_points_for_ellipse:
            continue
        
        # Try direct ellipse fitting
        coeffs = fit_ellipse_direct(cluster)
        
        if coeffs is not None:
            ellipse_params = ellipse_parameters(*coeffs)
            
            if ellipse_params is not None:
                # Validate ellipse
                a, b = ellipse_params['a'], ellipse_params['b']
                cx, cy = ellipse_params['center']
                
                # Check if ellipse is reasonable
                if (5 <= min(a, b) <= max(w, h) and
                    0 <= cx < w and 0 <= cy < h):
                    
                    # Calculate score based on preferences
                    score = 0
                    if prefer_largest:
                        score += min(a, b)  # Prefer larger minor axis
                    if prefer_highest_gradient:
                        # Calculate average gradient magnitude at edge points
                        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                        gradient_mag = np.sqrt(gx**2 + gy**2)
                        edge_gradients = [gradient_mag[int(y), int(x)] 
                                        for x, y in cluster 
                                        if 0 <= int(y) < h and 0 <= int(x) < w]
                        if len(edge_gradients) > 0:
                            score += np.mean(edge_gradients) * 0.1
                    
                    score += len(cluster) * 0.01  # Prefer clusters with more points
                    
                    detected_ellipses.append(ellipse_params)
                    ellipse_scores.append(score)
        
        # Fallback to circle fitting if ellipse failed and enabled
        elif use_circle_fallback and len(cluster) >= 3:
            try:
                (cx, cy), radius = cv2.minEnclosingCircle(cluster)
                cx, cy, radius = int(cx), int(cy), int(radius)
                
                if (5 <= radius <= max(w, h) and
                    0 <= cx < w and 0 <= cy < h):
                    
                    ellipse_params = {
                        'center': (cx, cy),
                        'axes': (radius * 2, radius * 2),
                        'angle': 0.0,
                        'xc': float(cx),
                        'yc': float(cy),
                        'a': float(radius),
                        'b': float(radius),
                        'orientation': 0.0,
                        'confidence': 0.8  # Lower confidence for circle fallback
                    }
                    
                    # Calculate score
                    score = radius * 0.5  # Prefer larger circles
                    if prefer_highest_gradient:
                        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                        gradient_mag = np.sqrt(gx**2 + gy**2)
                        edge_gradients = [gradient_mag[int(y), int(x)] 
                                        for x, y in cluster 
                                        if 0 <= int(y) < h and 0 <= int(x) < w]
                        if len(edge_gradients) > 0:
                            score += np.mean(edge_gradients) * 0.1
                    
                    detected_ellipses.append(ellipse_params)
                    ellipse_scores.append(score)
            except:
                pass
    
    # Step 5: Select best ellipse
    best_idx = None
    if len(detected_ellipses) > 0:
        if reference_point is None:
            reference_point = (w // 2, h // 2)
        
        # If we have scores, use them
        if len(ellipse_scores) == len(detected_ellipses):
            # Combine score with distance from reference
            for i, ellipse in enumerate(detected_ellipses):
                cx, cy = ellipse['center']
                dist = math.sqrt((cx - reference_point[0])**2 + (cy - reference_point[1])**2)
                # Prefer closer ellipses (inverse distance)
                ellipse_scores[i] += 100.0 / (dist + 1)
            
            best_idx = np.argmax(ellipse_scores)
        else:
            # Fallback: use distance from reference
            distances = []
            for ellipse in detected_ellipses:
                cx, cy = ellipse['center']
                dist = math.sqrt((cx - reference_point[0])**2 + (cy - reference_point[1])**2)
                distances.append(dist)
            best_idx = np.argmin(distances)
    
    # Step 6: Draw results
    result_img = img.copy()
    
    if draw_all_ellipses:
        # Draw all detected ellipses in blue
        for i, ellipse in enumerate(detected_ellipses):
            center = (int(ellipse['center'][0]), int(ellipse['center'][1]))
            axes = (int(ellipse['axes'][0]), int(ellipse['axes'][1]))
            angle = float(ellipse['angle'])
            color = (255, 0, 0) if i != best_idx else (0, 255, 0)
            cv2.ellipse(result_img, center, axes, angle, 0, 360, color, 2)
            cv2.circle(result_img, center, 5, color, -1)
    else:
        # Draw only best ellipse in green
        if best_idx is not None:
            ellipse = detected_ellipses[best_idx]
            center = (int(ellipse['center'][0]), int(ellipse['center'][1]))
            axes = (int(ellipse['axes'][0]), int(ellipse['axes'][1]))
            angle = float(ellipse['angle'])
            cv2.ellipse(result_img, center, axes, angle, 0, 360, (0, 255, 0), 3)
            cv2.circle(result_img, center, 5, (0, 255, 0), -1)
            
            # Draw big blue dot at determined center
            cv2.circle(result_img, center, 20, (255, 0, 0), -1)
            
            # Draw reference point
            if reference_point:
                cv2.circle(result_img, reference_point, 5, (0, 0, 255), -1)
    
    return result_img, detected_ellipses, best_idx, edges, mask


def process_video(video_path, output_dir, canny_edges_dir, **kwargs):
    """
    Process video file with Partial Ellipse Fit pipeline
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        **kwargs: Additional parameters for detect_hoops_partial_ellipse
    """
    video_path = Path(video_path)
    output_path = Path(output_dir)
    canny_path = Path(canny_edges_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    canny_path.mkdir(parents=True, exist_ok=True)
    
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
        
        result_img, ellipses, best_idx, edge_img, mask = detect_hoops_partial_ellipse(
            frame, **kwargs
        )
        
        output_file = output_path / f"frame_{frame_num:06d}_result.jpg"
        cv2.imwrite(str(output_file), result_img)
        
        canny_file = canny_path / f"frame_{frame_num:06d}_edges.jpg"
        cv2.imwrite(str(canny_file), edge_img)
        
        processed += 1
        frame_num += 1
        
        if processed % 10 == 0:
            print(f"  Processed {processed}/{total_frames} frames...")
            if len(ellipses) > 0:
                print(f"    Detected {len(ellipses)} ellipse(s), best: {best_idx}")
    
    cap.release()
    print(f"\n✓ Processed {processed} frames")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def process_frames(input_dir, output_dir, canny_edges_dir, num_frames=None, **kwargs):
    """
    Process frames with Partial Ellipse Fit pipeline
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        num_frames: Number of frames to process (None = process all available)
        **kwargs: Additional parameters for detect_hoops_partial_ellipse
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    canny_path = Path(canny_edges_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    canny_path.mkdir(parents=True, exist_ok=True)
    
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
        
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  Could not load {frame_name}, skipping...")
            continue
        
        result_img, ellipses, best_idx, edge_img, mask = detect_hoops_partial_ellipse(
            img, **kwargs
        )
        
        output_file = output_path / f"{frame_name.replace('.jpg', '_result.jpg')}"
        cv2.imwrite(str(output_file), result_img)
        
        canny_file = canny_path / f"{frame_name.replace('.jpg', '_edges.jpg')}"
        cv2.imwrite(str(canny_file), edge_img)
        
        processed += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
            if len(ellipses) > 0:
                print(f"    Detected {len(ellipses)} ellipse(s), best: {best_idx}")
    
    print(f"\n✓ Processed {processed} frames")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def process_test_images(test_images_dir, output_dir, canny_edges_dir, **kwargs):
    """
    Process test images with Partial Ellipse Fit pipeline
    
    Args:
        test_images_dir: Directory containing test images
        output_dir: Directory to save result images
        canny_edges_dir: Directory to save Canny edge images
        **kwargs: Additional parameters for detect_hoops_partial_ellipse
    """
    test_path = Path(test_images_dir)
    output_path = Path(output_dir)
    canny_path = Path(canny_edges_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    canny_path.mkdir(parents=True, exist_ok=True)
    
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
        
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  Could not load {img_file.name}, skipping...")
            continue
        
        result_img, ellipses, best_idx, edge_img, mask = detect_hoops_partial_ellipse(
            img, **kwargs
        )
        
        output_file = output_path / f"{img_file.stem}_result.jpg"
        cv2.imwrite(str(output_file), result_img)
        
        canny_file = canny_path / f"{img_file.stem}_edges.jpg"
        cv2.imwrite(str(canny_file), edge_img)
        
        if len(ellipses) > 0:
            print(f"  ✓ Detected {len(ellipses)} ellipse(s)")
            if best_idx is not None:
                ellipse = ellipses[best_idx]
                center = ellipse['center']
                axes = ellipse['axes']
                angle = ellipse['angle']
                print(f"    Best ellipse: center={center}, axes={axes}, angle={angle:.1f}°")
        else:
            print(f"  ✗ No ellipses detected")
        
        processed += 1
    
    print(f"\n✓ Processed {processed} test images")
    print(f"  Results saved to: {output_path}")
    print(f"  Canny edges saved to: {canny_path}")


def main():
    """Process test images with Partial Ellipse Fit pipeline"""
    test_images_dir = Path(__file__).parent.parent.parent.parent / "input" / "test_images"
    output_dir = Path(__file__).parent.parent / "output" / "test_results"
    canny_edges_dir = Path(__file__).parent.parent / "output" / "test_canny_edges"
    
    print(f"Processing test images with Partial Ellipse Fit pipeline...")
    print(f"Input: {test_images_dir}")
    print(f"Results: {output_dir}")
    print(f"Canny edges: {canny_edges_dir}\n")
    
    process_test_images(
        test_images_dir,
        output_dir,
        canny_edges_dir,
        use_red_mask=True,
        use_inner_edges=True,
        cluster_eps=20,  # Larger for bigger clusters
        cluster_min_samples=30,  # More points for robust fitting
        prefer_largest=True,
        prefer_highest_gradient=True,
        use_circle_fallback=True,
        draw_all_ellipses=False,
    )


if __name__ == "__main__":
    main()

