# Media Content Clipper

Extract audio clips from any media source based on quotes and timestamps. Supports media, YouTube videos, podcasts, and other online media.

## Project Structure
```
twitter-space-clipper/
├── run.sh               # Run the application
├── requirements.txt    # Python dependencies
├── app/               # Main application code
│   ├── __init__.py   # Marks directory as a Python package
│   ├── main.py       # Main Streamlit app with UI components
├── core/            # Core functionality
│   ├── __init__.py  # Marks directory as a Python package
│   ├── download.py  # Media download functionality
│   ├── transcribe.py # Audio transcription
│   ├── quotes.py    # Quote generation
│   ├── summary.py   # Summary generation
│   └── processor.py # Main processing pipeline
├── utils/          # Shared utilities
│   ├── __init__.py # Marks directory as a Python package
│   ├── api.py     # API utilities
│   └── file_utils.py # File handling utilities
├── .streamlit/    # Streamlit configuration
│   ├── config.toml  # Streamlit app configuration
│   └── secrets.toml # API keys and secrets (gitignored)
├── storage/      # Data storage (gitignored)
│   ├── downloads/    # Downloaded media
│   ├── transcripts/  # Generated transcripts
│   ├── quotes/      # Generated quotes
│   └── summaries/   # Generated summaries
└── docs/          # Documentation
```

### Understanding the Structure
- **Root Directory**: Contains user-facing scripts and configuration
  - `run.sh`: Start the application
  - `requirements.txt`: Python package dependencies
  
- **`app/`**: Main application code
  - Uses `__init__.py` to mark it as a Python package
  - Contains the Streamlit interface and UI components
  
- **`core/`**: Core business logic
  - Each module handles a specific functionality
  - Independent of the UI layer
  
- **`utils/`**: Shared utilities
  - Common functionality used across the application
  
- **`.streamlit/`**: Streamlit configuration
  - `config.toml`: App theme and server settings
  - `secrets.toml`: API keys and authentication (gitignored)
  
- **`storage/`**: Data storage
  - Organized by content type
  - Automatically created as needed

## Features
- Download media from various sources using yt-dlp
- Transcribe audio using OpenAI's Whisper
- Generate quotes from transcripts using DeepSeek AI
- Generate summaries using DeepSeek AI
- Web interface for easy processing and content management
- Password protection for the web interface
- History tracking for processed media

## Requirements
- Python 3.10 or higher
- ffmpeg for audio processing
- DeepSeek API credentials

## Installation

1. Set up the environment:
```bash
# Clone the repository
git clone <repository-url>
cd twitter-space-clipper

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg if not already installed
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html
```

2. Configure Streamlit:

Create `.streamlit/secrets.toml` with the following content:
```toml
# App password
password = "your_app_password"

# DeepSeek API credentials
DEEPSEEK_API_KEY = "your_deepseek_api_key"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/"
```

3. Run the application:
```bash
./run.sh  # On Windows: run.bat
```

## Usage

1. Open your browser to the displayed URL (typically http://localhost:8501)
2. Enter the password configured in `secrets.toml`
3. Paste a media URL or select from previously processed content
4. Wait for processing to complete
5. Access generated content:
   - Quotes with copy buttons
   - Full transcript
   - Audio download
   - Content summary
   - History of processed media

## Supported Media Sources
- YouTube videos
- Podcasts
- Any audio/video source supported by yt-dlp

## Core Modules
- `core/processor.py` - Main processing pipeline
- `core/download.py` - Media download functionality
- `core/transcribe.py` - Audio transcription
- `core/quotes.py` - Quote generation
- `core/summary.py` - Summary generation
- `celery_worker/` - Background job processing with Celery

## Celery Integration

The application uses Celery for handling long-running tasks like transcription, quote generation, and summary generation. This allows for better scalability and responsiveness of the web interface.

### Setup Redis (Broker/Backend)

1. Install Redis:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   
   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

2. Start Redis server:
   ```bash
   # Ubuntu/Debian
   sudo service redis-server start
   
   # macOS
   brew services start redis
   
   # Windows
   redis-server
   ```

3. Configure Redis connection in `.env`:
   ```
   CELERY_BROKER_URL=redis://localhost:6379/0
   CELERY_RESULT_BACKEND=redis://localhost:6379/0
   ```

### Running Celery Worker

1. Start the Celery worker:
   ```bash
   # From project root
   celery -A celery_worker.celery_app worker --loglevel=info
   ```

2. For development with auto-reload:
   ```bash
   # Install watchdog for auto-reload
   pip install watchdog
   
   # Start worker with auto-reload
   celery -A celery_worker.celery_app worker --loglevel=info --pool=solo --autoreload
   ```

### Task Status Monitoring

The application tracks task progress and status in the `storage/state/` directory. Each processing job has its own state file that includes:
- Current processing stage
- Progress percentage
- Status messages
- Error information (if any)

## Security Note
- The app is password protected via Streamlit secrets
- API keys are stored securely in `secrets.toml`
- Sensitive files are gitignored
- All user uploads are validated and sanitized
