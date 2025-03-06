# Twitter Space Clipper

Extract audio clips from Twitter Spaces based on quotes and timestamps.

## Features
- Download Twitter Space recordings
- Transcribe audio using OpenAI's Whisper
- Generate quotes from transcripts
- Create audio clips for specific quotes

## Requirements
- Python 3.11.x
- ffmpeg for audio processing
- Twitter API credentials
- DeepSeek API credentials

## Installation
1. Create a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```
TWITTER_API_KEY=your_key
TWITTER_API_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_TOKEN_SECRET=your_token_secret
DEEPSEEK_API_KEY=your_key
DEEPSEEK_API_URL=your_url
```

## Usage
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
- `xspace.py` - Download and process Twitter Spaces
- `xclips.py` - Create audio clips from quotes
- `xquotes.py` - Generate quotes from transcripts
- `transcribe.py` - Transcribe audio using Whisper
- `xdownload_space.py` - Download Twitter Space recordings
