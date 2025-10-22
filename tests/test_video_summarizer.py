"""
Test suite for the video summarization project.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import numpy as np
import cv2

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from video_summarizer import VideoSummarizer, create_summary_video
from utils import create_sample_video, get_video_info, validate_video_file
from config import Config


class TestVideoSummarizer(unittest.TestCase):
    """Test cases for VideoSummarizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_video = self.temp_dir / "test_video.mp4"
        
        # Create a sample video for testing
        create_sample_video(self.sample_video, duration=5, fps=10)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_histogram_method(self):
        """Test histogram-based summarization."""
        summarizer = VideoSummarizer(method="histogram", max_frames=5)
        summary_frames, indices = summarizer.summarize(self.sample_video)
        
        self.assertGreater(len(summary_frames), 0)
        self.assertLessEqual(len(summary_frames), 5)
        self.assertEqual(len(summary_frames), len(indices))
    
    def test_deep_learning_method(self):
        """Test deep learning-based summarization."""
        summarizer = VideoSummarizer(method="deep_learning", max_frames=5)
        summary_frames, indices = summarizer.summarize(self.sample_video)
        
        self.assertGreater(len(summary_frames), 0)
        self.assertLessEqual(len(summary_frames), 5)
        self.assertEqual(len(summary_frames), len(indices))
    
    def test_hybrid_method(self):
        """Test hybrid summarization method."""
        summarizer = VideoSummarizer(method="hybrid", max_frames=5)
        summary_frames, indices = summarizer.summarize(self.sample_video)
        
        self.assertGreater(len(summary_frames), 0)
        self.assertLessEqual(len(summary_frames), 5)
        self.assertEqual(len(summary_frames), len(indices))
    
    def test_save_summary_frames(self):
        """Test saving summary frames."""
        summarizer = VideoSummarizer(method="histogram", max_frames=3)
        summary_frames, _ = summarizer.summarize(self.sample_video)
        
        output_dir = self.temp_dir / "output"
        saved_paths = summarizer.save_summary_frames(summary_frames, output_dir)
        
        self.assertEqual(len(saved_paths), len(summary_frames))
        for path in saved_paths:
            self.assertTrue(path.exists())
    
    def test_create_summary_video(self):
        """Test creating summary video."""
        summarizer = VideoSummarizer(method="histogram", max_frames=3)
        summary_frames, _ = summarizer.summarize(self.sample_video)
        
        output_video = self.temp_dir / "summary.mp4"
        video_path = create_summary_video(summary_frames, output_video, fps=1)
        
        self.assertTrue(video_path.exists())
        self.assertEqual(video_path, output_video)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_sample_video(self):
        """Test creating sample video."""
        video_path = self.temp_dir / "sample.mp4"
        create_sample_video(video_path, duration=3, fps=5)
        
        self.assertTrue(video_path.exists())
        
        # Verify video properties
        info = get_video_info(video_path)
        self.assertGreater(info['frame_count'], 0)
        self.assertGreater(info['duration'], 0)
    
    def test_get_video_info(self):
        """Test getting video information."""
        video_path = self.temp_dir / "test.mp4"
        create_sample_video(video_path, duration=2, fps=10)
        
        info = get_video_info(video_path)
        
        self.assertIn('fps', info)
        self.assertIn('frame_count', info)
        self.assertIn('width', info)
        self.assertIn('height', info)
        self.assertIn('duration', info)
        
        self.assertGreater(info['fps'], 0)
        self.assertGreater(info['frame_count'], 0)
        self.assertGreater(info['width'], 0)
        self.assertGreater(info['height'], 0)
        self.assertGreater(info['duration'], 0)
    
    def test_validate_video_file(self):
        """Test video file validation."""
        # Test with valid video
        video_path = self.temp_dir / "valid.mp4"
        create_sample_video(video_path)
        self.assertTrue(validate_video_file(video_path))
        
        # Test with non-existent file
        non_existent = self.temp_dir / "nonexistent.mp4"
        self.assertFalse(validate_video_file(non_existent))
        
        # Test with non-video file
        text_file = self.temp_dir / "test.txt"
        text_file.write_text("This is not a video")
        self.assertFalse(validate_video_file(text_file))


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        
        # Test getting values
        self.assertEqual(config.get("video_summarizer.scene_threshold"), 0.5)
        self.assertEqual(config.get("video_summarizer.max_frames"), 20)
        self.assertEqual(config.get("video_summarizer.method"), "hybrid")
        
        # Test getting non-existent key
        self.assertIsNone(config.get("non.existent.key"))
        self.assertEqual(config.get("non.existent.key", "default"), "default")
    
    def test_set_config(self):
        """Test setting configuration values."""
        config = Config()
        
        config.set("video_summarizer.scene_threshold", 0.7)
        self.assertEqual(config.get("video_summarizer.scene_threshold"), 0.7)
        
        config.set("new.section.value", "test")
        self.assertEqual(config.get("new.section.value"), "test")
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        config = Config()
        config.set("test.value", "test_data")
        
        # Save config
        config_path = self.temp_dir / "test_config.yaml"
        config.save(config_path)
        self.assertTrue(config_path.exists())
        
        # Load config
        new_config = Config(config_path)
        self.assertEqual(new_config.get("test.value"), "test_data")


if __name__ == "__main__":
    unittest.main()
