"""
Script to process video with Ellipse-YOLO pipeline
"""

from pathlib import Path
from detect_hoops_ellipse_yolo import process_video

if __name__ == "__main__":
    # Paths
    video_path = Path(__file__).parent.parent.parent.parent / "input" / "front.mp4"
    output_dir = Path(__file__).parent.parent / "output" / "video_results"
    canny_edges_dir = Path(__file__).parent.parent / "output" / "video_canny_edges"
    output_video_path = Path(__file__).parent.parent / "output" / "ellipse_yolo_video.mp4"
    
    print(f"Processing video with Ellipse-YOLO pipeline...")
    print(f"Input video: {video_path}")
    print(f"Results: {output_dir}")
    print(f"Canny edges: {canny_edges_dir}")
    print(f"Output video: {output_video_path}\n")
    
    process_video(
        video_path,
        output_dir,
        canny_edges_dir,
        output_video_path=output_video_path,  # Create output video
        use_red_mask=True,  # Use red masking
        use_model=True,  # Try to use model if available
        model_path=None,  # Set to model path if you have trained model
        use_inner_edges=True,  # Focus on inner edges only
        use_partial_circles=True,  # Support partial circles
        draw_all_ellipses=False,  # Only draw nearest ellipse
        fallback_maxRadius=None,  # Adaptive based on image size
    )

