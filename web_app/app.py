"""
Streamlit web interface for video summarization.

This module provides a user-friendly web interface for the video summarization
toolkit using Streamlit.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging
from typing import List, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from video_summarizer import VideoSummarizer, create_summary_video
from utils import setup_logging, visualize_summary, get_video_info, validate_video_file, create_sample_video
from config import Config

# Setup logging
setup_logging()

# Page configuration
st.set_page_config(
    page_title="Video Summarization Tool",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Video Summarization Tool</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Method selection
        method = st.selectbox(
            "Summarization Method",
            ["histogram", "deep_learning", "hybrid"],
            help="Choose the method for video summarization"
        )
        
        # Scene threshold
        scene_threshold = st.slider(
            "Scene Change Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher values detect fewer scene changes"
        )
        
        # Max frames
        max_frames = st.slider(
            "Maximum Summary Frames",
            min_value=5,
            max_value=50,
            value=20,
            help="Maximum number of frames in the summary"
        )
        
        # Device selection
        device = st.selectbox(
            "Processing Device",
            ["auto", "cpu", "cuda", "mps"],
            help="Device for deep learning processing"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to summarize"
        )
        
        # Sample video option
        if st.button("🎥 Generate Sample Video"):
            with st.spinner("Creating sample video..."):
                sample_path = Path("data/sample_video.mp4")
                sample_path.parent.mkdir(exist_ok=True)
                create_sample_video(sample_path, duration=15, fps=30)
                st.success("Sample video created!")
                st.session_state.sample_video = str(sample_path)
        
        # Display sample video info
        if hasattr(st.session_state, 'sample_video'):
            st.info(f"Sample video available: {st.session_state.sample_video}")
    
    with col2:
        st.header("📊 Video Information")
        
        video_path = None
        if uploaded_file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = Path(tmp_file.name)
        elif hasattr(st.session_state, 'sample_video'):
            video_path = Path(st.session_state.sample_video)
        
        if video_path and video_path.exists():
            try:
                info = get_video_info(video_path)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Duration", f"{info['duration']:.1f} seconds")
                st.metric("FPS", f"{info['fps']:.1f}")
                st.metric("Resolution", f"{info['width']}x{info['height']}")
                st.metric("Total Frames", f"{info['frame_count']:,}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error reading video info: {e}")
    
    # Processing section
    if video_path and video_path.exists():
        st.header("🔄 Processing")
        
        if st.button("🚀 Generate Summary", type="primary"):
            with st.spinner("Processing video..."):
                try:
                    # Initialize summarizer
                    summarizer = VideoSummarizer(
                        scene_threshold=scene_threshold,
                        max_frames=max_frames,
                        method=method,
                        device=device
                    )
                    
                    # Generate summary
                    summary_frames, selected_indices = summarizer.summarize(video_path)
                    
                    # Store results in session state
                    st.session_state.summary_frames = summary_frames
                    st.session_state.selected_indices = selected_indices
                    st.session_state.video_path = str(video_path)
                    
                    st.success(f"✅ Summary generated with {len(summary_frames)} frames!")
                    
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    logging.error(f"Processing error: {e}")
    
    # Results section
    if hasattr(st.session_state, 'summary_frames') and st.session_state.summary_frames:
        st.header("📋 Results")
        
        summary_frames = st.session_state.summary_frames
        selected_indices = st.session_state.selected_indices
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Summary Frames", len(summary_frames))
        
        with col2:
            compression_ratio = len(summary_frames) / len(summary_frames) * 100 if summary_frames else 0
            st.metric("Compression Ratio", f"{compression_ratio:.1f}%")
        
        with col3:
            if hasattr(st.session_state, 'video_path'):
                try:
                    info = get_video_info(Path(st.session_state.video_path))
                    duration_reduction = (1 - len(summary_frames) / info['frame_count']) * 100
                    st.metric("Duration Reduction", f"{duration_reduction:.1f}%")
                except:
                    st.metric("Duration Reduction", "N/A")
        
        with col4:
            st.metric("Method Used", method.title())
        
        # Display summary frames
        st.subheader("🎞️ Summary Frames")
        
        # Create a grid of frames
        cols_per_row = 5
        for i in range(0, len(summary_frames), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                frame_idx = i + j
                if frame_idx < len(summary_frames):
                    frame = summary_frames[frame_idx]
                    # Convert BGR to RGB for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    col.image(frame_rgb, caption=f"Frame {frame_idx + 1}", use_column_width=True)
        
        # Download options
        st.subheader("💾 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Download Summary Images"):
                # Create zip file with summary frames
                import zipfile
                import io
                
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for i, frame in enumerate(summary_frames):
                        # Encode frame as JPEG
                        _, buffer = cv2.imencode('.jpg', frame)
                        zip_file.writestr(f"summary_frame_{i+1:03d}.jpg", buffer.tobytes())
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📦 Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="video_summary_frames.zip",
                    mime="application/zip"
                )
        
        with col2:
            if st.button("🎬 Create Summary Video"):
                with st.spinner("Creating summary video..."):
                    try:
                        output_path = Path("output/summary_video.mp4")
                        output_path.parent.mkdir(exist_ok=True)
                        
                        summary_video_path = create_summary_video(
                            summary_frames, 
                            output_path, 
                            fps=2
                        )
                        
                        # Read video file for download
                        with open(summary_video_path, 'rb') as f:
                            video_data = f.read()
                        
                        st.download_button(
                            label="📹 Download Summary Video",
                            data=video_data,
                            file_name="video_summary.mp4",
                            mime="video/mp4"
                        )
                        
                    except Exception as e:
                        st.error(f"Error creating summary video: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Video Summarization Tool v1.0.0 | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
