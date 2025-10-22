"""
Utility functions for the video summarization project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def create_sample_video(
    output_path: Path, 
    duration: int = 10, 
    fps: int = 30,
    width: int = 640,
    height: int = 480
) -> Path:
    """
    Create a sample video with multiple scenes for testing.
    
    Args:
        output_path: Path for output video
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        
    Returns:
        Path to created video
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration * fps
    frames_per_scene = total_frames // 5  # 5 different scenes
    
    try:
        for frame_idx in range(total_frames):
            # Create different scenes
            scene_idx = frame_idx // frames_per_scene
            
            if scene_idx == 0:
                # Red scene
                frame = np.full((height, width, 3), (0, 0, 255), dtype=np.uint8)
            elif scene_idx == 1:
                # Green scene
                frame = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)
            elif scene_idx == 2:
                # Blue scene
                frame = np.full((height, width, 3), (255, 0, 0), dtype=np.uint8)
            elif scene_idx == 3:
                # Gradient scene
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    intensity = int(255 * y / height)
                    frame[y, :] = [intensity, intensity, intensity]
            else:
                # Random noise scene
                frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            
            # Add frame number text
            cv2.putText(
                frame, 
                f"Frame {frame_idx}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
            
            out.write(frame)
    finally:
        out.release()
    
    logging.info(f"Created sample video: {output_path}")
    return output_path


def visualize_summary(
    summary_frames: list, 
    output_path: Optional[Path] = None,
    max_display: int = 10
) -> None:
    """
    Visualize summary frames in a grid layout.
    
    Args:
        summary_frames: List of frames to display
        output_path: Optional path to save visualization
        max_display: Maximum number of frames to display
    """
    if not summary_frames:
        logging.warning("No frames to visualize")
        return
    
    # Limit frames for display
    display_frames = summary_frames[:max_display]
    n_frames = len(display_frames)
    
    # Calculate grid dimensions
    cols = min(5, n_frames)
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, frame in enumerate(display_frames):
        if i < len(axes):
            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(frame_rgb)
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_frames, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Video Summary - {n_frames} Key Frames", fontsize=16)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved visualization to {output_path}")
    
    plt.show()


def get_video_info(video_path: Path) -> dict:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    try:
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        return info
    finally:
        cap.release()


def validate_video_file(video_path: Path) -> bool:
    """
    Validate if a file is a valid video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid video file, False otherwise
    """
    if not video_path.exists():
        return False
    
    cap = cv2.VideoCapture(str(video_path))
    is_valid = cap.isOpened()
    cap.release()
    
    return is_valid
