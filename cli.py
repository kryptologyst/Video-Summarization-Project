#!/usr/bin/env python3
"""
Command-line interface for video summarization.

This module provides a command-line interface for the video summarization toolkit.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from video_summarizer import VideoSummarizer, create_summary_video
from utils import setup_logging, visualize_summary, get_video_info, validate_video_file
from config import Config


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Video Summarization Tool - Extract key frames from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py input_video.mp4
  
  # Specify method and parameters
  python cli.py input_video.mp4 --method hybrid --max-frames 15 --threshold 0.3
  
  # Save summary as video
  python cli.py input_video.mp4 --output-video summary.mp4
  
  # Verbose output
  python cli.py input_video.mp4 --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        "input_video",
        type=str,
        help="Path to input video file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--method",
        choices=["histogram", "deep_learning", "hybrid"],
        default="hybrid",
        help="Summarization method (default: hybrid)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Scene change threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=20,
        help="Maximum number of summary frames (default: 20)"
    )
    
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Processing device (default: auto)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for summary frames (default: output)"
    )
    
    parser.add_argument(
        "--output-video",
        type=str,
        help="Path for output summary video"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="FPS for output video (default: 2)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display summary frames visualization"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration if provided
    config = None
    if args.config:
        config = Config(Path(args.config))
        logger.info(f"Loaded configuration from {args.config}")
    
    # Validate input video
    input_path = Path(args.input_video)
    if not input_path.exists():
        logger.error(f"Input video file not found: {input_path}")
        sys.exit(1)
    
    if not validate_video_file(input_path):
        logger.error(f"Invalid video file: {input_path}")
        sys.exit(1)
    
    try:
        # Get video information
        logger.info("Analyzing input video...")
        video_info = get_video_info(input_path)
        logger.info(f"Video info: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['duration']:.1f}s, {video_info['frame_count']} frames")
        
        # Initialize summarizer
        logger.info(f"Initializing summarizer with method: {args.method}")
        summarizer = VideoSummarizer(
            scene_threshold=args.threshold,
            max_frames=args.max_frames,
            method=args.method,
            device=args.device
        )
        
        # Generate summary
        logger.info("Generating video summary...")
        summary_frames, selected_indices = summarizer.summarize(input_path)
        
        logger.info(f"Summary generated with {len(summary_frames)} frames")
        logger.info(f"Selected frame indices: {selected_indices}")
        
        # Save summary frames
        output_dir = Path(args.output_dir)
        saved_paths = summarizer.save_summary_frames(summary_frames, output_dir)
        logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        
        # Create summary video if requested
        if args.output_video:
            logger.info("Creating summary video...")
            video_path = create_summary_video(
                summary_frames, 
                Path(args.output_video), 
                fps=args.fps
            )
            logger.info(f"Summary video saved to: {video_path}")
        
        # Display visualization if requested
        if args.visualize:
            logger.info("Displaying visualization...")
            visualize_summary(summary_frames)
        
        # Print summary statistics
        compression_ratio = len(summary_frames) / video_info['frame_count'] * 100
        duration_reduction = (1 - len(summary_frames) / video_info['frame_count']) * 100
        
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Original frames: {video_info['frame_count']:,}")
        print(f"Summary frames: {len(summary_frames)}")
        print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"Duration reduction: {duration_reduction:.1f}%")
        print(f"Method used: {args.method}")
        print(f"Output directory: {output_dir}")
        if args.output_video:
            print(f"Summary video: {args.output_video}")
        print("="*50)
        
        logger.info("Video summarization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
