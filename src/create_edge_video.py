"""
Create Video from Canny Edge Images
Combines all edge images into a video file
"""

import cv2
from pathlib import Path
import glob


def create_video_from_edges(edges_dir, output_video_path, fps=30):
    """
    Create a video from Canny edge images
    
    Args:
        edges_dir: Directory containing edge images
        output_video_path: Path to save the output video
        fps: Frames per second for the video (default: 30)
    """
    edges_path = Path(edges_dir)
    output_path = Path(output_video_path)
    
    # Get all edge image files, sorted by frame number
    edge_files = sorted(edges_path.glob("frame_*_edges.jpg"))
    
    if not edge_files:
        print(f"Error: No edge images found in {edges_path}")
        return
    
    print(f"Found {len(edge_files)} edge images")
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(edge_files[0]))
    if first_img is None:
        print(f"Error: Could not read first image {edge_files[0]}")
        return
    
    height, width, channels = first_img.shape
    print(f"Video dimensions: {width}x{height}")
    print(f"Frame rate: {fps} fps")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create video file {output_path}")
        return
    
    # Write all frames
    for i, edge_file in enumerate(edge_files):
        img = cv2.imread(str(edge_file))
        if img is None:
            print(f"Warning: Could not read {edge_file}, skipping...")
            continue
        
        # Ensure image matches video dimensions (resize if needed)
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(edge_files)} frames...")
    
    out.release()
    print(f"\n✓ Video created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Total frames: {len(edge_files)}")


def main():
    """Create video from Canny edge images"""
    edges_dir = Path(__file__).parent.parent / "output" / "canny_edges"
    output_video = Path(__file__).parent.parent / "output" / "canny_edges_video.mp4"
    
    print(f"Creating video from Canny edge images...")
    print(f"Input: {edges_dir}")
    print(f"Output: {output_video}\n")
    
    create_video_from_edges(edges_dir, output_video, fps=30)


if __name__ == "__main__":
    main()


