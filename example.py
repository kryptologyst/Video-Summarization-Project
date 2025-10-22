#!/usr/bin/env python3
"""
Example script demonstrating video summarization usage.

This script shows how to use the video summarization toolkit with different
methods and configurations.
"""

import sys
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from video_summarizer import VideoSummarizer, create_summary_video
from utils import setup_logging, create_sample_video, visualize_summary
from config import Config

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting video summarization example")
    
    # Create sample video if it doesn't exist
    sample_video = Path("data/sample_video.mp4")
    sample_video.parent.mkdir(exist_ok=True)
    
    if not sample_video.exists():
        logger.info("Creating sample video...")
        create_sample_video(sample_video, duration=10, fps=30)
    
    # Load configuration
    config = Config()
    
    # Example 1: Histogram method
    logger.info("Example 1: Histogram-based summarization")
    summarizer_hist = VideoSummarizer(
        method="histogram",
        scene_threshold=0.3,
        max_frames=10
    )
    
    summary_frames_hist, indices_hist = summarizer_hist.summarize(sample_video)
    logger.info(f"Histogram method selected {len(summary_frames_hist)} frames")
    
    # Example 2: Deep learning method
    logger.info("Example 2: Deep learning-based summarization")
    summarizer_dl = VideoSummarizer(
        method="deep_learning",
        max_frames=8,
        device="auto"
    )
    
    try:
        summary_frames_dl, indices_dl = summarizer_dl.summarize(sample_video)
        logger.info(f"Deep learning method selected {len(summary_frames_dl)} frames")
    except Exception as e:
        logger.warning(f"Deep learning method failed: {e}")
        logger.info("Falling back to histogram method")
        summary_frames_dl, indices_dl = summary_frames_hist, indices_hist
    
    # Example 3: Hybrid method
    logger.info("Example 3: Hybrid summarization")
    summarizer_hybrid = VideoSummarizer(
        method="hybrid",
        scene_threshold=0.4,
        max_frames=12
    )
    
    summary_frames_hybrid, indices_hybrid = summarizer_hybrid.summarize(sample_video)
    logger.info(f"Hybrid method selected {len(summary_frames_hybrid)} frames")
    
    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Save histogram method results
    summarizer_hist.save_summary_frames(summary_frames_hist, output_dir / "histogram")
    
    # Save deep learning method results
    summarizer_dl.save_summary_frames(summary_frames_dl, output_dir / "deep_learning")
    
    # Save hybrid method results
    summarizer_hybrid.save_summary_frames(summary_frames_hybrid, output_dir / "hybrid")
    
    # Create summary videos
    logger.info("Creating summary videos...")
    
    create_summary_video(
        summary_frames_hist, 
        output_dir / "histogram_summary.mp4", 
        fps=2
    )
    
    create_summary_video(
        summary_frames_dl, 
        output_dir / "deep_learning_summary.mp4", 
        fps=2
    )
    
    create_summary_video(
        summary_frames_hybrid, 
        output_dir / "hybrid_summary.mp4", 
        fps=2
    )
    
    # Display visualizations
    logger.info("Displaying visualizations...")
    
    visualize_summary(summary_frames_hist, output_dir / "histogram_visualization.png")
    visualize_summary(summary_frames_dl, output_dir / "deep_learning_visualization.png")
    visualize_summary(summary_frames_hybrid, output_dir / "hybrid_visualization.png")
    
    # Print summary
    print("\n" + "="*60)
    print("VIDEO SUMMARIZATION EXAMPLE COMPLETED")
    print("="*60)
    print(f"Input video: {sample_video}")
    print(f"Histogram method: {len(summary_frames_hist)} frames")
    print(f"Deep learning method: {len(summary_frames_dl)} frames")
    print(f"Hybrid method: {len(summary_frames_hybrid)} frames")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
