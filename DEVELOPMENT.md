# Video Summarization Project - Development Setup

This document provides instructions for setting up the development environment for the video summarization project.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd video-summarization
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Run tests**:
   ```bash
   python -m pytest tests/
   ```

5. **Run the example**:
   ```bash
   python example.py
   ```

6. **Launch the web interface**:
   ```bash
   streamlit run web_app/app.py
   ```

## Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

Run all quality checks:
```bash
black src/ tests/ web_app/ cli.py example.py
flake8 src/ tests/ web_app/ cli.py example.py
mypy src/
isort src/ tests/ web_app/ cli.py example.py
```

## Testing

Run the full test suite:
```bash
python -m pytest tests/ -v
```

Run tests with coverage:
```bash
python -m pytest tests/ --cov=src --cov-report=html
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
├── example.py             # Example usage script
├── requirements.txt       # Dependencies
├── requirements-dev.txt   # Development dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Main documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run quality checks: `black`, `flake8`, `mypy`, `isort`
6. Run tests: `python -m pytest tests/`
7. Commit your changes: `git commit -m "Add feature"`
8. Push to the branch: `git push origin feature-name`
9. Submit a pull request

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the project root directory
2. **CUDA errors**: Install PyTorch with CUDA support or use CPU-only version
3. **Model download issues**: Check internet connection for Hugging Face models
4. **Memory issues**: Reduce `max_frames` parameter or use smaller models

### Debug Mode

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the CLI with verbose flag:
```bash
python cli.py input_video.mp4 --verbose
```
