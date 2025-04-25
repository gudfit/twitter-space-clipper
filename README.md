# Media Content Clipper

Extract audio clips from any media source based on quotes and timestamps. Supports media, YouTube videos, podcasts, and other online media.

## Features
- Download media from various sources using yt-dlp
- Transcribe audio using OpenAI's Whisper
- Generate quotes from transcripts using DeepSeek AI
- Create audio clips for specific quotes
- Web interface for easy processing and quote management

## Requirements
- Python 3.10 or higher
- ffmpeg for audio processing
- DeepSeek API credentials

## Installation
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.streamlit/secrets.toml`:
```toml
# Authentication
password = "your_password"  # Password for web interface access

# DeepSeek API credentials
DEEPSEEK_API_KEY = "your_key"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
```

Note: The `.streamlit/secrets.toml` file is required for the web interface. For command line usage only, you can alternatively use a `.env` file with the same DeepSeek API credentials.

## Usage

### Web Interface (Recommended)
Run the Streamlit app:
```bash
./run.sh  # On Windows: run.bat
```
Then open your browser to the displayed URL (typically http://localhost:8501)

### Command Line Interface
For advanced users, you can also use the command line tools:

1. Download and transcribe media:
```bash
python xspace.py "URL_TO_MEDIA"
```

2. Extract audio clips from quotes:
```bash
python xclips.py path/to/quotes.txt [--audio audio_file] [--output output_dir]
```

The clips will be saved in a 'clips' subdirectory by default.

## Supported Media Sources
- media
- YouTube videos
- Podcasts
- Any audio/video source supported by yt-dlp

## Scripts
- `app.py` - Streamlit web interface
- `xspace.py` - Download and process media content
- `xclips.py` - Create audio clips from quotes
- `xquotes.py` - Generate quotes from transcripts using DeepSeek AI
- `transcribe.py` - Transcribe audio using Whisper
- `xdownload_space.py` - Download media using yt-dlp
