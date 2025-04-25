# Twitter Space Clipper

Extract audio clips from Twitter Spaces based on quotes and timestamps.

## Features
- Download Twitter Space recordings using yt-dlp
- Transcribe audio using OpenAI's Whisper
- Generate quotes from transcripts using DeepSeek AI
- Create audio clips for specific quotes
- Web interface for easy processing and quote management

## Requirements
- Python 3.11.x
- ffmpeg for audio processing
- DeepSeek API credentials

## Installation
1. Create a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
DEEPSEEK_API_KEY=your_key
DEEPSEEK_API_URL=your_url
```

## Usage

### Web Interface (Recommended)
Run the Streamlit app:
```bash
./run.sh  # On Windows: run.bat
```
Then open your browser to the displayed URL (typically http://localhost:8501)

### Command Line Interface
For advanced users, you can also use the command line tools:

1. Download and transcribe a Twitter Space:
```bash
python xspace.py "https://twitter.com/i/spaces/your-space-id"
```

2. Extract audio clips from quotes:
```bash
python xclips.py path/to/quotes.txt [--audio audio_file] [--output output_dir]
```

The clips will be saved in a 'clips' subdirectory by default.

## Scripts
- `app.py` - Streamlit web interface
- `xspace.py` - Download and process Twitter Spaces
- `xclips.py` - Create audio clips from quotes
- `xquotes.py` - Generate quotes from transcripts using DeepSeek AI
- `transcribe.py` - Transcribe audio using Whisper
- `xdownload_space.py` - Download Twitter Space recordings using yt-dlp
