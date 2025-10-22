# Video Summarization Project

A modern video summarization toolkit that combines traditional computer vision techniques with state-of-the-art deep learning methods for creating concise video summaries.

## 🚀 Quick Start

### Web Interface (Recommended)
```bash
streamlit run web_app/app.py
```

### Command Line
```bash
python cli.py input_video.mp4 --method hybrid --max-frames 20
```

### Python API
```python
from src.video_summarizer import VideoSummarizer

summarizer = VideoSummarizer(method="hybrid", max_frames=20)
summary_frames, indices = summarizer.summarize("input_video.mp4")
```

## 📋 Features

- **Multiple Methods**: Histogram, deep learning, and hybrid approaches
- **Modern AI**: Hugging Face transformers integration
- **User-Friendly**: Both CLI and web interfaces
- **Configurable**: YAML/JSON configuration support
- **Type-Safe**: Full type hints and PEP8 compliance
- **Tested**: Comprehensive test suite

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 📖 Documentation

- [README.md](README.md) - Main documentation
- [DEVELOPMENT.md](DEVELOPMENT.md) - Development setup
- [Example Usage](example.py) - Example script

## 🧪 Testing

```bash
python -m pytest tests/
```

## 📁 Project Structure

```
├── src/                    # Source code
├── web_app/               # Streamlit interface
├── tests/                 # Test suite
├── config/                # Configuration
├── data/                  # Data directory
├── output/                # Output directory
├── cli.py                 # CLI interface
└── example.py             # Example usage
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
