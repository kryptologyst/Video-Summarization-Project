"""
Core video summarization functionality.

This module provides both traditional computer vision-based summarization
and modern deep learning approaches for video summarization.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class VideoSummarizer:
    """
    A comprehensive video summarization class that supports multiple methods.
    
    This class provides both traditional histogram-based scene detection and
    modern deep learning approaches for video summarization.
    """
    
    def __init__(
        self,
        scene_threshold: float = 0.5,
        max_frames: int = 20,
        method: str = "histogram",
        device: str = "auto"
    ):
        """
        Initialize the VideoSummarizer.
        
        Args:
            scene_threshold: Threshold for scene change detection (0.0-1.0)
            max_frames: Maximum number of frames to include in summary
            method: Summarization method ("histogram", "deep_learning", "hybrid")
            device: Device to use for deep learning ("auto", "cpu", "cuda")
        """
        self.scene_threshold = scene_threshold
        self.max_frames = max_frames
        self.method = method
        self.device = self._setup_device(device)
        
        # Initialize deep learning components if needed
        self.tokenizer = None
        self.model = None
        if method in ["deep_learning", "hybrid"]:
            self._setup_deep_learning()
    
    def _setup_device(self, device: str) -> str:
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _setup_deep_learning(self) -> None:
        """Setup deep learning models for advanced summarization."""
        try:
            logger.info("Loading deep learning models...")
            # Using a lightweight vision transformer for feature extraction
            model_name = "google/vit-base-patch16-224"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Deep learning models loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load deep learning models: {e}")
            logger.info("Falling back to histogram method")
            self.method = "histogram"
    
    def extract_frames(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of extracted frames
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frames.append(frame.copy())
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from {video_path.name}")
        return frames
    
    def histogram_scene_detection(self, frames: List[np.ndarray]) -> List[int]:
        """
        Detect scene changes using histogram comparison.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of frame indices where scenes change
        """
        scene_indices = [0]  # Always include first frame
        prev_hist = None
        
        for i, frame in enumerate(frames):
            # Convert to HSV and compute histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > self.scene_threshold:
                    scene_indices.append(i)
            
            prev_hist = hist
        
        logger.info(f"Detected {len(scene_indices)} scene changes")
        return scene_indices
    
    def deep_learning_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract deep learning features from frames.
        
        Args:
            frames: List of video frames
            
        Returns:
            Feature matrix of shape (n_frames, feature_dim)
        """
        if self.model is None:
            raise ValueError("Deep learning model not initialized")
        
        features = []
        
        with torch.no_grad():
            for frame in frames:
                # Preprocess frame for ViT
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frame_tensor = torch.from_numpy(frame_resized).float()
                frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
                frame_tensor = frame_tensor.to(self.device)
                
                # Extract features
                outputs = self.model(frame_tensor)
                features.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
        
        return np.vstack(features)
    
    def cluster_based_selection(self, features: np.ndarray) -> List[int]:
        """
        Select representative frames using clustering.
        
        Args:
            features: Feature matrix
            
        Returns:
            List of selected frame indices
        """
        n_clusters = min(self.max_frames, len(features))
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Select frames closest to cluster centers
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_features) > 0:
                # Find frame closest to cluster center
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        selected_indices.sort()
        logger.info(f"Selected {len(selected_indices)} frames using clustering")
        return selected_indices
    
    def summarize(self, video_path: Union[str, Path]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate video summary using the specified method.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (summary_frames, selected_indices)
        """
        logger.info(f"Starting video summarization using {self.method} method")
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        if self.method == "histogram":
            selected_indices = self.histogram_scene_detection(frames)
        elif self.method == "deep_learning":
            features = self.deep_learning_features(frames)
            selected_indices = self.cluster_based_selection(features)
        elif self.method == "hybrid":
            # Combine both methods
            hist_indices = self.histogram_scene_detection(frames)
            features = self.deep_learning_features(frames)
            cluster_indices = self.cluster_based_selection(features)
            
            # Merge and deduplicate indices
            all_indices = sorted(set(hist_indices + cluster_indices))
            selected_indices = all_indices[:self.max_frames]
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Limit to max_frames
        selected_indices = selected_indices[:self.max_frames]
        summary_frames = [frames[i] for i in selected_indices]
        
        logger.info(f"Generated summary with {len(summary_frames)} frames")
        return summary_frames, selected_indices
    
    def save_summary_frames(
        self, 
        summary_frames: List[np.ndarray], 
        output_dir: Union[str, Path] = "output"
    ) -> List[Path]:
        """
        Save summary frames to disk.
        
        Args:
            summary_frames: List of frames to save
            output_dir: Directory to save frames
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        saved_paths = []
        for i, frame in enumerate(summary_frames):
            filename = f"summary_frame_{i+1:03d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_paths.append(filepath)
        
        logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths


def create_summary_video(
    summary_frames: List[np.ndarray], 
    output_path: Union[str, Path],
    fps: int = 2
) -> Path:
    """
    Create a summary video from selected frames.
    
    Args:
        summary_frames: List of frames to include in video
        output_path: Path for output video file
        fps: Frames per second for output video
        
    Returns:
        Path to created video file
    """
    output_path = Path(output_path)
    
    if not summary_frames:
        raise ValueError("No frames provided for video creation")
    
    # Get frame dimensions
    height, width = summary_frames[0].shape[:2]
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    try:
        for frame in summary_frames:
            out.write(frame)
    finally:
        out.release()
    
    logger.info(f"Created summary video: {output_path}")
    return output_path
