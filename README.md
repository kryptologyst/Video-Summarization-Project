# Video Summarization Project

A modern video summarization toolkit that combines traditional computer vision techniques with state-of-the-art deep learning methods for creating concise video summaries.

## Features

- **Multiple Summarization Methods**: Histogram-based scene detection, deep learning features, and hybrid approaches
- **Modern Deep Learning**: Integration with Hugging Face transformers for advanced feature extraction
- **User-Friendly Interfaces**: Both CLI and Streamlit web interface
- **Comprehensive Testing**: Full test suite with synthetic data generation
- **Configurable**: YAML/JSON configuration support
- **Type-Safe**: Full type hints and PEP8 compliance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Video-Summarization-Project.git
cd Video-Summarization-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web interface:
```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` and upload a video file.

### Command Line Interface

Basic usage:
```bash
python cli.py input_video.mp4
```

Advanced usage:
```bash
python cli.py input_video.mp4 --method hybrid --max-frames 15 --threshold 0.3 --output-video summary.mp4
```

### Python API

```python
from src.video_summarizer import VideoSummarizer

# Initialize summarizer
summarizer = VideoSummarizer(
    method="hybrid",
    max_frames=20,
    scene_threshold=0.5
)

# Generate summary
summary_frames, indices = summarizer.summarize("input_video.mp4")

# Save frames
summarizer.save_summary_frames(summary_frames, "output/")
```

## Methods

### 1. Histogram Method
- Uses HSV histogram comparison to detect scene changes
- Fast and lightweight
- Good for videos with distinct color changes

### 2. Deep Learning Method
- Uses Vision Transformer (ViT) for feature extraction
- Clusters frames based on visual similarity
- More sophisticated but requires more computational resources

### 3. Hybrid Method
- Combines both histogram and deep learning approaches
- Provides the best balance of speed and quality
- Recommended for most use cases

## Configuration

Create a `config/default.yaml` file to customize settings:

```yaml
video_summarizer:
  scene_threshold: 0.5
  max_frames: 20
  method: hybrid
  device: auto

output:
  directory: output
  video_fps: 2
  image_format: jpg

deep_learning:
  model_name: google/vit-base-patch16-224
  batch_size: 1
```

## Project Structure

```
video-summarization/
├── src/                    # Source code
│   ├── __init__.py
│   ├── video_summarizer.py # Core summarization logic
│   ├── config.py          # Configuration management
│   └── utils.py           # Utility functions
├── web_app/               # Streamlit web interface
│   └── app.py
├── tests/                 # Test suite
│   └── test_video_summarizer.py
├── config/                # Configuration files
│   └── default.yaml
├── data/                  # Data directory
├── models/                # Model storage
├── output/                # Output directory
├── cli.py                 # Command-line interface
├── requirements.txt       # Dependencies
├── requirements-dev.txt   # Development dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Examples

### Basic Video Summarization
```python
from src.video_summarizer import VideoSummarizer

summarizer = VideoSummarizer()
summary_frames, indices = summarizer.summarize("video.mp4")
```

### Custom Configuration
```python
from src.config import Config
from src.video_summarizer import VideoSummarizer

config = Config("config/custom.yaml")
summarizer = VideoSummarizer(
    scene_threshold=config.get("video_summarizer.scene_threshold"),
    max_frames=config.get("video_summarizer.max_frames"),
    method=config.get("video_summarizer.method")
)
```

### Creating Summary Video
```python
from src.video_summarizer import create_summary_video

summary_video_path = create_summary_video(
    summary_frames, 
    "output/summary.mp4", 
    fps=2
)
```

## Performance Tips

1. **Device Selection**: Use `cuda` for GPU acceleration if available
2. **Method Choice**: Use `histogram` for speed, `deep_learning` for quality
3. **Frame Limits**: Adjust `max_frames` based on your needs
4. **Threshold Tuning**: Lower thresholds detect more scene changes

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `max_frames` or use `cpu` device
2. **Model Download Issues**: Check internet connection for Hugging Face models
3. **Video Format Issues**: Ensure video file is in a supported format (mp4, avi, mov, mkv)

### Debug Mode

Enable verbose logging:
```bash
python cli.py input_video.mp4 --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- Hugging Face for transformer models
- Streamlit for the web interface
- The computer vision community for research and techniques
# Video-Summarization-Project
