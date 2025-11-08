"""
Canny Edge Detection on Video
Processes video and saves only the Canny edge images
"""

import cv2
from pathlib import Path


def process_video(video_path, output_dir, canny_low=50, canny_high=150):
    """
    Process video file directly and save Canny edge images
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save edge images
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
    """
    video_path = Path(video_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video has {total_frames} frames")
    
    processed = 0
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Save edges
        output_file = output_path / f"frame_{frame_num:06d}_edges.jpg"
        cv2.imwrite(str(output_file), edges)
        
        processed += 1
        frame_num += 1
        
        if processed % 10 == 0:
            print(f"  Processed {processed}/{total_frames} frames...")
    
    cap.release()
    print(f"\n✓ Processed {processed} frames")
    print(f"  Output saved to: {output_path}")


def process_frames(input_dir, output_dir, num_frames=None, canny_low=50, canny_high=150):
    """
    Process frames and save only Canny edge images
    
    Args:
        input_dir: Directory containing input frames
        output_dir: Directory to save edge images
        num_frames: Number of frames to process (None = process all available)
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # If num_frames is None, find all available frames
    if num_frames is None:
        frame_files = sorted(input_path.glob("frame_*.jpg"))
        if frame_files:
            # Extract frame numbers and find max
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
        # Format frame number with zero padding
        frame_name = f"frame_{i:06d}.jpg"
        frame_path = input_path / frame_name
        
        if not frame_path.exists():
            # Stop if we hit a gap (frame doesn't exist)
            if processed > 0:
                break
            continue
        
        # Load image
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  Could not load {frame_name}, skipping...")
            continue
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # Save only the edges
        output_file = output_path / f"{frame_name.replace('.jpg', '_edges.jpg')}"
        cv2.imwrite(str(output_file), edges)
        
        processed += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
    
    print(f"\n✓ Processed {processed} frames")
    print(f"  Output saved to: {output_path}")


def main():
    """Process all frames from video"""
    video_path = Path(__file__).parent.parent.parent / "input" / "front.mp4"
    output_dir = Path(__file__).parent.parent / "output" / "canny_edges"
    
    print(f"Processing entire video...")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Only saving Canny edge images...\n")
    
    process_video(video_path, output_dir)


if __name__ == "__main__":
    main()

