import streamlit as st  # type: ignore
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import json
import hashlib
import shutil
import yt_dlp  # type: ignore
import re
from typing import Optional, Dict, Any, List, Tuple, Callable, TextIO, BinaryIO, Protocol, Union, cast
import time
import logging
from streamlit_extras.stylable_container import stylable_container  # type: ignore
from datetime import datetime, timedelta

from core.types import ProcessState, StoragePaths

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add file handler with more detailed format for debug logs
debug_handler = logging.FileHandler('app.debug.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(debug_handler)

logger.info("Starting application initialization...")

# Disable watchdog for PyTorch modules to prevent custom class errors
import streamlit.watcher.path_watcher  # type: ignore

original_watch_file = streamlit.watcher.path_watcher.watch_file

def patched_watch_file(filepath: str, *args: Any, **kwargs: Any) -> Optional[bool]:
    """Patched watch_file function that ignores PyTorch files."""
    if 'torch' in filepath or '_C' in filepath:
        return None
    return cast(bool, original_watch_file(filepath, *args, **kwargs))

# Type ignore because streamlit's type stubs don't match implementation
streamlit.watcher.path_watcher.watch_file = patched_watch_file  # type: ignore

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    logger.debug("Checking ffmpeg installation...")
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        logger.debug("ffmpeg check successful")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("ffmpeg not found")
        return False

# Configure Streamlit page
logger.info("Configuring Streamlit page...")
st.set_page_config(
    page_title="LinkToQuotes",
    page_icon="🎙️",
    layout="wide"
)

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    logger.debug("Checking password authentication...")
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            logger.info("Password authentication successful")
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            logger.warning("Incorrect password attempt")
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        logger.debug("First password attempt")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        logger.debug("Previous password attempt failed")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        logger.debug("Password already verified")
        return True

# Only import core functionality after password check
if check_password():
    logger.info("Password verified, loading core functionality...")
    
    # Import core functionality
    from core.download import download_twitter_space
    from core.transcribe import transcribe_audio
    from core.quotes import chunk_transcript
    from core.summary import generate_summary, save_summary, load_summary
    from core.processor import (
        process_space,
        get_space_id,
        get_storage_paths,
        regenerate_quotes,
        get_process_state,
        save_process_state,
        ProcessLock,
        ProgressCallback
    )
    from utils.api import call_deepseek_api

    # Check for ffmpeg at startup
    if not check_ffmpeg():
        logger.error("ffmpeg not found, stopping application")
        st.error("""
        ⚠️ ffmpeg is not installed or not found in PATH. This is required for processing media.
        
        To install ffmpeg:
        - Ubuntu/Debian: `sudo apt-get install ffmpeg`
        - macOS: `brew install ffmpeg`
        - Windows: Download from https://ffmpeg.org/download.html
        
        After installing, please restart the application.
        """)
        st.stop()

    logger.info("Initializing session state...")
    # Initialize session state for progress tracking
    if not hasattr(st.session_state, 'processing_complete'):
        setattr(st.session_state, 'processing_complete', False)
    if not hasattr(st.session_state, 'current_space_id'):
        setattr(st.session_state, 'current_space_id', None)
    if not hasattr(st.session_state, 'download_progress'):
        setattr(st.session_state, 'download_progress', 0.0)
    if not hasattr(st.session_state, 'total_fragments'):
        setattr(st.session_state, 'total_fragments', 0)
    if not hasattr(st.session_state, 'current_fragment'):
        setattr(st.session_state, 'current_fragment', 0)
    if not hasattr(st.session_state, 'regenerating_quotes'):
        setattr(st.session_state, 'regenerating_quotes', False)
    if not hasattr(st.session_state, 'selected_media'):
        setattr(st.session_state, 'selected_media', None)
    if not hasattr(st.session_state, 'url_history'):
        setattr(st.session_state, 'url_history', {})
    if not hasattr(st.session_state, 'loaded_space_id'):
        setattr(st.session_state, 'loaded_space_id', None)

    logger.info("Creating storage directories...")
    # Create persistent storage directories
    STORAGE_DIR = Path("storage")
    STORAGE_DIR.mkdir(exist_ok=True)
    DOWNLOADS_DIR = STORAGE_DIR / "downloads"
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    QUOTES_DIR = STORAGE_DIR / "quotes"
    QUOTES_DIR.mkdir(exist_ok=True)
    SUMMARIES_DIR = STORAGE_DIR / "summaries"
    SUMMARIES_DIR.mkdir(exist_ok=True)

    # Create URL history file
    URL_HISTORY_FILE = STORAGE_DIR / "url_history.json"
    if not URL_HISTORY_FILE.exists():
        logger.info("Creating new URL history file")
        with open(URL_HISTORY_FILE, 'w') as f:
            json.dump({}, f)

    def log_processing_step(step: str, status: str = "started", details: Optional[str] = None) -> None:
        """Helper function to log processing steps consistently"""
        message = f"{step} {status}"
        if details:
            message += f": {details}"
        logger.info(message)

    def load_url_history():
        """Load URL history from file."""
        try:
            with open(URL_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def save_url_history(space_id: str, url: str):
        """Save URL to history."""
        history = load_url_history()
        history[space_id] = url
        with open(URL_HISTORY_FILE, 'w') as f:
            json.dump(history, f)
        st.session_state.url_history = history

    def get_url_from_history(space_id: str) -> str:
        """Get original URL for a space_id."""
        if not hasattr(st.session_state, 'url_history'):
            st.session_state.url_history = load_url_history()
        return st.session_state.url_history.get(space_id, "Unknown URL")

    def read_quotes(quotes_path: str) -> List[str]:
        """Read quotes from a file and return as a list.
        
        Args:
            quotes_path: Path to the quotes file
            
        Returns:
            List of quotes, or empty list if file doesn't exist
        """
        try:
            with open(quotes_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [q.strip() for q in content.split('\n\n') if q.strip()]
        except Exception as e:
            logger.error(f"Error reading quotes: {e}")
            return []

    def display_quotes(quotes: List[str], container: Any):
        """Display quotes in a Streamlit container with copy buttons.
        
        Args:
            quotes: List of quotes to display
            container: Streamlit container to display in
        """
        for i, quote in enumerate(quotes, 1):
            with container:                
                with stylable_container(
                    key=f"quote_{i}",
                    css_styles="""
                        code {
                            white-space: pre-wrap !important;
                        }
                        """
                ):
                    st.code(quote, language="text")

    class ProgressCallbackImpl(Protocol):
        """Protocol for progress tracking callbacks."""
        def __call__(self, stage: str, progress: float, status: str) -> None: ...

    class StreamlitProgressCallback:
        """Progress callback that updates Streamlit UI."""
        def __init__(self, container: Any):
            self.container = container
            self.progress_bar = None
            
        def __call__(self, stage: str, progress: float, status: str) -> None:
            if not self.progress_bar:
                self.progress_bar = self.container.progress(0)
            
            # Update progress bar with emoji indicators for each stage
            stage_emoji = {
                "download": "⬇️",
                "transcribe": "🎯",
                "quotes": "✍️",
                "summary": "📝",
                "complete": "✅",
                "error": "❌"
            }
            
            emoji = stage_emoji.get(stage, "")
            # Update progress bar with emoji
            if self.progress_bar is not None:
                self.progress_bar.progress(progress, f"{emoji} {stage.title()}: {status}")
            
            # Only show completion/error messages outside progress bar
            if stage == "complete":
                self.container.success("✅ Processing complete!")
            elif stage == "error":
                self.container.error(f"❌ Error: {status}")

    def check_process_state(space_id: str) -> ProcessState:
        """Check the current state of processing for a space."""
        state = get_process_state(str(STORAGE_DIR), space_id)
        
        # Check if process is running but stale
        if state['status'] == 'processing':
            last_updated_str = state.get('last_updated')
            if last_updated_str:
                last_updated = datetime.fromisoformat(last_updated_str)
                time_since_update = datetime.now() - last_updated
                logger.debug(f"Process state last updated: {time_since_update} ago")
                
                # Only mark as error if significantly stale (> 1 hour)
                if time_since_update > timedelta(hours=1):
                    logger.warning(f"Process {space_id} appears stale (no updates for {time_since_update})")
                    state.update({
                        'status': 'error',
                        'error': 'Process timed out - no updates for over an hour'
                    })
                    save_process_state(str(STORAGE_DIR), space_id, state)
                else:
                    logger.info(f"Process {space_id} is still active (last update: {time_since_update} ago)")
        
        return state

    def display_process_state(state: ProcessState, container: Any) -> None:
        """Display the current process state in the UI."""
        status = state['status']
        files = state.get('files', {})
        stage = state.get('stage')
        stage_status = state.get('stage_status', '')
        
        with container:
            if status == 'processing':
                # Show stage-specific progress information
                progress = state['progress']
                
                # Create progress display based on stage
                if stage == 'download':
                    st.info("⬇️ Downloading Media")
                    st.progress(progress)
                    st.caption(str(stage_status))  # Show detailed status (speed, etc.)
                elif stage == 'transcribe':
                    st.info("🎯 Transcribing Audio")
                    st.progress(progress)
                    
                    # Show transcription details
                    last_updated_str = state.get('last_updated')
                    if last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str)
                        time_since = datetime.now() - last_updated
                        
                        # Show elapsed time with appropriate units
                        if time_since < timedelta(minutes=1):
                            elapsed = f"{time_since.seconds} seconds"
                        elif time_since < timedelta(hours=1):
                            elapsed = f"{time_since.seconds // 60} minutes"
                        else:
                            elapsed = f"{time_since.seconds // 3600} hours"
                        
                        # Show file being transcribed
                        if files.get('audio', False):
                            paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                            audio_path = paths['audio_path']
                            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Convert to MB
                            st.caption(f"🎵 Transcribing: {Path(audio_path).name} ({file_size:.1f} MB)")
                        
                        # Show elapsed time and model info
                        st.caption(f"⏱️ Elapsed time: {elapsed}")
                        st.caption("🤖 Using OpenAI Whisper base model")
                    
                    st.caption(str(stage_status))
                elif stage == 'quotes':
                    st.info("✍️ Generating Quotes")
                    st.progress(progress)
                    
                    # Show quote generation details
                    last_updated_str = state.get('last_updated')
                    if last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str)
                        time_since = datetime.now() - last_updated
                        
                        # Show elapsed time with appropriate units
                        if time_since < timedelta(minutes=1):
                            elapsed = f"{time_since.seconds} seconds"
                        elif time_since < timedelta(hours=1):
                            elapsed = f"{time_since.seconds // 60} minutes"
                        else:
                            elapsed = f"{time_since.seconds // 3600} hours"
                        
                        # Show transcript details
                        if files.get('transcript', False):
                            paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                            transcript_path = paths['transcript_path']
                            with open(transcript_path, 'r', encoding='utf-8') as f:
                                transcript = f.read()
                            
                            # Get chunk information
                            chunks = chunk_transcript(transcript)
                            total_chunks = len(chunks)
                            current_chunk = int(progress * total_chunks) if progress > 0 else 0
                            
                            # Show chunk progress
                            st.caption(f"📊 Processing chunk {current_chunk}/{total_chunks}")
                            st.caption(f"📝 Total transcript length: {len(transcript):,} characters")
                        
                        # Show elapsed time and model info
                        st.caption(f"⏱️ Elapsed time: {elapsed}")
                        st.caption("🤖 Using DeepSeek for quote extraction")
                    
                    st.caption(str(stage_status))
                elif stage == 'summary':
                    st.info("📝 Creating Summary")
                    st.progress(progress)
                    
                    # Show summary generation details
                    last_updated_str = state.get('last_updated')
                    if last_updated_str:
                        last_updated = datetime.fromisoformat(last_updated_str)
                        time_since = datetime.now() - last_updated
                        
                        # Show elapsed time with appropriate units
                        if time_since < timedelta(minutes=1):
                            elapsed = f"{time_since.seconds} seconds"
                        elif time_since < timedelta(hours=1):
                            elapsed = f"{time_since.seconds // 60} minutes"
                        else:
                            elapsed = f"{time_since.seconds // 3600} hours"
                        
                        # Show input details
                        if files.get('transcript', False) and files.get('quotes', False):
                            paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                            
                            # Show transcript info
                            with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                                transcript = f.read()
                            st.caption(f"📝 Processing transcript: {len(transcript):,} characters")
                            
                            # Show quotes info
                            quotes = read_quotes(paths['quotes_path'])
                            st.caption(f"✍️ Using {len(quotes)} generated quotes")
                        
                        # Show elapsed time and model info
                        st.caption(f"⏱️ Elapsed time: {elapsed}")
                        st.caption("�� Using DeepSeek for summarization")
                        
                        # Show stages of summary generation
                        if progress < 0.3:
                            st.caption("📊 Analyzing transcript...")
                        elif progress < 0.6:
                            st.caption("📝 Extracting key points...")
                        elif progress < 0.9:
                            st.caption("✨ Generating overview...")
                        else:
                            st.caption("💾 Saving summary...")
                    
                    st.caption(str(stage_status))
                else:
                    stage_display = stage if stage else "Unknown"
                    st.info(f"🔄 Processing: {stage_display} ({progress*100:.0f}%)")
                    st.progress(progress)
                
                # Show file status
                if files:
                    st.markdown("#### File Status")
                    cols = st.columns([1, 1, 1, 1])
                    file_types = ['audio', 'transcript', 'quotes', 'summary']
                    for i, file_type in enumerate(file_types):
                        with cols[i]:
                            exists = files.get(file_type, False)
                            icon = "✅" if exists else "❌"
                            st.markdown(f"**{icon} {file_type.title()}**")
                
                # Show last update time if available
                last_updated_str = state.get('last_updated')
                if last_updated_str and stage not in ['transcribe', 'quotes', 'summary']:  # Don't show twice for these stages
                    last_updated = datetime.fromisoformat(last_updated_str)
                    time_since = datetime.now() - last_updated
                    if time_since < timedelta(minutes=1):
                        st.caption(f"Last updated: {time_since.seconds} seconds ago")
                    elif time_since < timedelta(hours=1):
                        st.caption(f"Last updated: {time_since.seconds // 60} minutes ago")
                    else:
                        st.caption(f"Last updated: {time_since.seconds // 3600} hours ago")
                
            elif status == 'error':
                error_msg = state.get('error', 'Unknown error')
                st.error(f"❌ Error: {error_msg}")
                
                # Show file status if any files exist
                if any(files.values()):
                    st.markdown("#### Existing Files")
                    cols = st.columns([1, 1, 1, 1])
                    file_types = ['audio', 'transcript', 'quotes', 'summary']
                    for i, file_type in enumerate(file_types):
                        with cols[i]:
                            exists = files.get(file_type, False)
                            if exists:
                                st.markdown(f"✅ {file_type.title()}")
                
                if st.button("🔄 Retry Processing"):
                    # Clear error state but preserve file status
                    state.update({
                        'status': 'not_started',
                        'error': None,
                        'progress': 0.0
                    })
                    save_process_state(str(STORAGE_DIR), space_id, state)
                    st.rerun()
                
            elif status == 'complete':
                if all(files.values()):
                    st.success("✅ Processing complete!")
                else:
                    # Some files are missing despite complete status
                    missing = [file_type for file_type in ['audio', 'transcript', 'quotes', 'summary'] 
                              if not files.get(file_type, False)]
                    st.warning(f"⚠️ Processing marked complete but missing files: {', '.join(missing)}")
                    if st.button("🔄 Reprocess Missing Files"):
                        state.update({
                            'status': 'not_started',
                            'error': None,
                            'progress': 0.0
                        })
                        save_process_state(str(STORAGE_DIR), space_id, state)
                        st.rerun()

    def process_space_with_ui(url: str, _progress_container: Any) -> Optional[StoragePaths]:
        """Process media URL with Streamlit UI updates."""
        try:
            log_processing_step("Space processing", "started", f"URL: {url}")
            space_id = get_space_id(url)
            
            # Check if already being processed
            state = check_process_state(space_id)
            if state['status'] == 'processing':
                display_process_state(state, _progress_container)
                st.info("⏳ This URL is already being processed. You can view its progress in the sidebar.")
                return None
            
            # Check if all files already exist
            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
            if all(os.path.exists(str(p)) for p in storage_paths.values()):
                st.success("✅ All files already exist for this URL!")
                st.session_state.processing_complete = True
                return storage_paths
            
            # Create progress callback
            progress_callback = StreamlitProgressCallback(_progress_container)
            
            # Process the space
            result_paths = process_space(url, str(STORAGE_DIR), progress_callback)
            
            if result_paths:
                st.session_state.processing_complete = True
                return result_paths
            else:
                # Check final state for error message
                state = check_process_state(space_id)
                if state['error']:
                    st.error(f"Processing failed: {state['error']}")
                else:
                    st.error("Processing failed. Check the logs for details.")
                return None
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None

    def download_with_progress(url: str, output_dir: str, progress_bar: Any) -> Optional[str]:
        """Download media with progress tracking."""
        space_id = get_space_id(url)
        # Create a lock file path
        lock_file = os.path.join(output_dir, f"{space_id}.lock")
        
        try:
            # Try to acquire a lock for this download
            with ProcessLock(str(STORAGE_DIR), space_id, timeout=3600):
                # First check if MP3 already exists
                final_mp3 = os.path.join(output_dir, f"{space_id}.mp3")
                if os.path.exists(final_mp3):
                    logger.info(f"MP3 file already exists: {final_mp3}")
                    progress_bar.progress(1.0, "File already exists")
                    return final_mp3
                
                # First download without post-processing
                download_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
                    'progress_hooks': [lambda d: progress_bar.progress(d['downloaded_bytes'] / d['total_bytes'] if d['total_bytes'] else 0)],
                    'keepvideo': True,  # Keep the original file
                    'postprocessors': [],  # No post-processing yet
                }
                
                try:
                    with yt_dlp.YoutubeDL(download_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                        original_file = os.path.join(output_dir, f"{info['id']}.{info['ext']}")
                        
                        # Convert to MP3 if needed
                        if info['ext'] != 'mp3':
                            if check_ffmpeg():
                                logger.info("Converting to MP3...")
                                progress_bar.progress(0.9, "Converting to MP3...")
                                
                                # Run ffmpeg conversion
                                os.system(f'ffmpeg -i "{original_file}" -vn -ar 44100 -ac 2 -b:a 192k "{final_mp3}"')
                                
                                # Clean up original file
                                os.remove(original_file)
                                
                                progress_bar.progress(1.0, "Download complete")
                                return final_mp3
                            else:
                                logger.warning("ffmpeg not found - keeping original format")
                                progress_bar.progress(1.0, "Download complete (original format)")
                                return original_file
                        
                        # If already MP3, just return the original file
                        return original_file
                        
                except Exception as e:
                    logger.error(f"Download error: {str(e)}")
                    progress_bar.progress(1.0, f"Error: {str(e)}")
                    raise
                
        except TimeoutError:
            logger.error("Timeout waiting for download lock")
            st.error("Another download is in progress. Please wait.")
            return None
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            st.error(f"Download error: {str(e)}")
            return None

    def get_space_id(url: str) -> str:
        """Extract space ID from URL and create a hash for storage."""
        space_id = url.strip('/').split('/')[-1]
        return hashlib.md5(space_id.encode()).hexdigest()

    def get_storage_paths(storage_root: str, space_id: str) -> StoragePaths:
        """Get paths for storing space data."""
        return {
            'audio_path': str(DOWNLOADS_DIR / f"{space_id}.mp3"),
            'transcript_path': str(TRANSCRIPTS_DIR / f"{space_id}.txt"),
            'quotes_path': str(QUOTES_DIR / f"{space_id}.txt"),
            'summary_path': str(SUMMARIES_DIR / f"{space_id}.json")
        }

    def find_processing_spaces() -> List[Tuple[str, ProcessState, str]]:
        """Find all spaces currently being processed."""
        logger.info("Starting find_processing_spaces scan...")
        processing_spaces: List[Tuple[str, ProcessState, str]] = []
        state_dir = STORAGE_DIR / "state"
        
        if not state_dir.exists():
            logger.debug("State directory does not exist")
            return []
        
        try:
            # Look through all state files
            state_files = list(state_dir.glob("*.json"))
            logger.debug(f"Found {len(state_files)} state files")
            
            for state_file in state_files:
                try:
                    logger.debug(f"Checking state file: {state_file}")
                    space_id = state_file.stem
                    state = get_process_state(str(STORAGE_DIR), space_id)
                    
                    # Check if process is running or recently updated
                    if state['status'] in ['processing', 'error']:
                        last_updated_str = state.get('last_updated')
                        if last_updated_str:
                            last_updated = datetime.fromisoformat(last_updated_str)
                            time_since_update = datetime.now() - last_updated
                            logger.debug(f"Time since last update: {time_since_update}")
                            
                            # Consider process active if updated in last hour or has error status
                            if time_since_update <= timedelta(hours=1) or state['status'] == 'error':
                                # Get original URL if available
                                original_url = get_url_from_history(space_id)
                                logger.debug(f"Adding active process: {space_id} - {original_url}")
                                processing_spaces.append((space_id, state, original_url))
                            else:
                                logger.debug(f"Skipping stale process: {space_id}")
                except Exception as e:
                    logger.error(f"Error checking state file {state_file}: {e}")
                    continue
                
            logger.info(f"Found {len(processing_spaces)} active processing spaces")
            return processing_spaces
            
        except Exception as e:
            logger.error(f"Error in find_processing_spaces: {e}")
            return []

    def load_previous_media(space_id: str) -> Optional[StoragePaths]:
        """Load previously processed media files."""
        storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
        if all(os.path.exists(str(p)) for p in storage_paths.values()):
            st.session_state.loaded_space_id = space_id
            st.session_state.processing_complete = True
            return storage_paths
        return None

    # Main app
    logger.info("Starting main application UI...")
    st.title("🎙️ LinkToQuote")
    st.write("Generate quotes and clips from any media url -  no listening required!")

    # Add persistent status indicator
    logger.debug("Checking for active processes...")
    processing_spaces = find_processing_spaces()
    if processing_spaces:
        with st.sidebar:
            st.markdown("### 🔄 Active Processes")
            for space_id, state, original_url in processing_spaces:
                with st.container():
                    if original_url:
                        st.info(f"🔗 Processing: {original_url[:50]}...")
                    else:
                        st.info(f"🔄 Processing space: {space_id[:8]}")
                    stage_display = state['stage'] if state['stage'] is not None else "Unknown"
                    st.progress(state['progress'], f"{stage_display.title()}: {state['progress']*100:.0f}%")
                    if st.button("📂 View Details", key=f"view_{space_id}_sidebar"):
                        st.session_state.current_space_id = space_id
                        st.session_state.processing_complete = False
                        st.rerun()
            st.markdown("---")

    # Create tabs for main content and sidebar content
    main_tab, summary_tab, logs_tab, history_tab, help_tab = st.tabs(["🎯 Main", "📝 Summary", "🔍 Logs", "📚 History", "❓ Help"])

    with main_tab:
        # Load existing media files and their URLs
        media_files = list(DOWNLOADS_DIR.glob("*.mp3")) if DOWNLOADS_DIR.exists() else []
        url_history = load_url_history()
        
        # Create two columns for input methods
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # URL input
            space_url = st.text_input("Paste Media URL:", placeholder="https://x.com/i/status/1915404626754494957")
            if space_url:  # If URL is pasted, automatically select "New URL"
                st.session_state.selected_option = "New URL"
        
        with col2:
            if media_files:
                # Create dropdown for previous URLs
                options = ["New URL"]
                options.extend([f"{url_history.get(Path(f).stem, 'Unknown URL')} - {Path(f).stem}" 
                              for f in media_files])
                
                selected_option = st.selectbox(
                    "Or select previous:",
                    options,
                    index=0 if space_url else None,  # Default to "New URL" if URL is pasted
                    key="selected_option",  # Add key to track state
                    help="Select a previously processed media file",
                    format_func=lambda x: x if x == "New URL" else f"{x.split(' - ')[0]} - {x.split(' - ')[1][:8]}"  # Only clip hash for display
                )
                
                if selected_option and selected_option != "New URL":  # Check for None and "New URL"
                    space_id = selected_option.split(" - ")[1]  # Get the full hash
                    space_url = url_history.get(space_id, "")
                    if st.button("📂 Load Selected"):
                        paths = load_previous_media(space_id)
                        if paths:
                            st.rerun()

        # Handle URL input or loaded media
        if space_url or st.session_state.loaded_space_id:
            # Get space ID with proper type checking
            current_id = get_space_id(space_url) if space_url else st.session_state.loaded_space_id
            if not isinstance(current_id, str):
                st.error("Invalid space ID")
                st.stop()
            
            space_id = current_id  # Now space_id is definitely str
            state = check_process_state(space_id)
            
            # Save URL to history when processing new media
            if space_url and space_id != st.session_state.current_space_id:
                save_url_history(space_id, space_url)
                st.session_state.current_space_id = space_id
                
                # Only reset state if not already processing
                if state['status'] != 'processing':
                    st.session_state.processing_complete = False
                    st.session_state.download_progress = 0
                    st.session_state.total_fragments = 0
                    st.session_state.current_fragment = 0

            if not st.session_state.processing_complete and state['status'] != 'complete':
                progress_container = st.container()
                with st.status("Processing media...", expanded=True) as status:
                    try:
                        # Display current state if processing
                        if state['status'] == 'processing':
                            display_process_state(state, progress_container)
                        else:
                            process_result = process_space_with_ui(space_url, progress_container)
                            if process_result:
                                st.session_state.processing_complete = True
                                st.success("✅ Space processed successfully!")
                                
                                # Check if we need to generate summary
                                if st.session_state.current_space_id is not None:
                                    storage_paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
                                    if not os.path.exists(storage_paths['summary_path']):
                                        if os.path.exists(storage_paths['transcript_path']) and os.path.exists(storage_paths['quotes_path']):
                                            with st.spinner("Generating summary..."):
                                                with open(storage_paths['transcript_path'], 'r', encoding='utf-8') as f:
                                                    transcript = f.read()
                                                quotes = read_quotes(storage_paths['quotes_path'])
                                                
                                                summary = generate_summary(transcript, quotes, storage_paths['summary_path'])
                                                
                                                if summary['overview'] != "Error generating summary":
                                                    save_summary(summary, storage_paths['summary_path'])
                                                    st.success("✨ Summary generated successfully!")
                                                else:
                                                    st.error("Failed to generate summary. Please try manually.")
                    except Exception as e:
                        st.error(f"Error processing Space: {str(e)}")
                        status.update(label="Error!", state="error")
                        st.session_state.processing_complete = False

            # If processing is complete or media was loaded, show results
            if st.session_state.processing_complete:
                storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                
                # Show original URL for loaded content
                if st.session_state.loaded_space_id:
                    original_url = get_url_from_history(space_id)
                    st.info(f"🔗 Loaded content from: {original_url}")

                # Display tabs for different outputs
                content_tab1, content_tab2, content_tab3 = st.tabs(["📝 Quotes", "🎵 Audio", "📄 Transcript"])
                
                with content_tab1:
                    st.subheader("Generated Quotes")
                    if os.path.exists(storage_paths['quotes_path']):
                        # Add regenerate button at the top
                        col1, col2 = st.columns([4, 1])
                        with col2:
                            if st.button("Regenerate", key="regenerate_quotes"):
                                st.session_state.regenerating_quotes = True
                                
                                # Create containers for progress and logs
                                status_container = st.empty()
                                progress_container = st.empty()
                                log_container = st.empty()
                                
                                with st.spinner("Regenerating quotes..."):
                                    # Create progress bar
                                    progress_bar = progress_container.progress(0)
                                    
                                    # Read transcript
                                    status_container.info("Reading transcript...")
                                    progress_bar.progress(0.2)
                                    
                                    try:
                                        # Create a log expander
                                        with log_container.expander("Generation Logs", expanded=True):
                                            st.write("Starting quote generation...")
                                            
                                            # Capture stdout to show logs
                                            import io
                                            import sys
                                            old_stdout = sys.stdout
                                            sys.stdout = mystdout = io.StringIO()
                                            
                                            try:
                                                quotes = regenerate_quotes(
                                                    storage_paths['transcript_path'],
                                                    storage_paths['quotes_path'],
                                                    space_url
                                                )
                                                
                                                # Get the logs
                                                logs = mystdout.getvalue()
                                                with stylable_container(
                                                    key="generation_logs",
                                                    css_styles="""
                                                        code {
                                                            white-space: pre-wrap !important;
                                                        }
                                                        """
                                                ):
                                                    st.code(logs)
                                                
                                                if quotes:
                                                    status_container.success("✨ Quotes regenerated successfully!")
                                                    progress_bar.progress(1.0)
                                                else:
                                                    status_container.error("❌ Failed to generate quotes. Check the logs for details.")
                                                    progress_bar.progress(1.0)
                                                    
                                            finally:
                                                sys.stdout = old_stdout
                                    
                                    except Exception as e:
                                        status_container.error(f"❌ Error: {str(e)}")
                                        progress_bar.progress(1.0)
                                    
                                st.session_state.regenerating_quotes = False
                                time.sleep(1)  # Give time for status messages to be read
                                st.rerun()
                        
                        # Read and display quotes in the main container
                        quotes = read_quotes(storage_paths['quotes_path'])
                        if quotes:  # Only try to display if we have quotes
                            display_quotes(quotes, st.container())
                        else:
                            st.warning("No quotes found in the file. Try regenerating the quotes.")
                
                with content_tab2:
                    st.subheader("Audio")
                    if os.path.exists(storage_paths['audio_path']):
                        audio_size = os.path.getsize(storage_paths['audio_path']) / (1024 * 1024)  # MB
                        st.write(f"Audio file size: {audio_size:.1f} MB")
                        
                        with open(storage_paths['audio_path'], 'rb') as audio_file:
                            audio_data = audio_file.read()
                            st.download_button(
                                "⬇️ Download Full Recording",
                                audio_data,
                                file_name=f"space_{space_id[:8]}.mp3",
                                mime="audio/mpeg"
                            )
                
                with content_tab3:
                    st.subheader("Transcript")
                    if os.path.exists(storage_paths['transcript_path']):
                        with open(storage_paths['transcript_path'], 'r') as f:
                            transcript = f.read()
                        
                        # Split transcript into chunks
                        chunks = chunk_transcript(transcript)
                        total_chunks = len(chunks)
                        
                        # Add chunk navigation
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"Transcript length: {len(transcript):,} characters ({total_chunks} pages)")
                        with col2:
                            chunk_idx = st.number_input("Page", min_value=1, max_value=total_chunks, value=1) - 1
                        
                        # Show current chunk with some context
                        st.text_area(
                            f"Transcript (Page {chunk_idx + 1}/{total_chunks})", 
                            chunks[chunk_idx],
                            height=400
                        )
                        
                        # Navigation buttons
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col1:
                            if chunk_idx > 0:
                                if st.button("⬅️ Previous Page"):
                                    st.session_state.chunk_idx = max(0, chunk_idx - 1)
                                    st.rerun()
                        with col3:
                            if chunk_idx < total_chunks - 1:
                                if st.button("Next Page ➡️"):
                                    st.session_state.chunk_idx = min(total_chunks - 1, chunk_idx + 1)
                                    st.rerun()
                        
                        # Download button for full transcript
                        st.download_button(
                            "⬇️ Download Full Transcript",
                            transcript,
                            file_name=f"transcript_{space_id[:8]}.txt",
                            mime="text/plain"
                        )

    with summary_tab:
        st.subheader("📝 Content Summary")
        
        if st.session_state.processing_complete and st.session_state.current_space_id:
            storage_paths = get_storage_paths(str(STORAGE_DIR), st.session_state.current_space_id)
            
            # Check for required files first
            files_status = {
                'transcript': os.path.exists(storage_paths['transcript_path']),
                'quotes': os.path.exists(storage_paths['quotes_path']),
                'summary': os.path.exists(storage_paths['summary_path'])
            }
            
            if not files_status['transcript'] or not files_status['quotes']:
                st.warning("⚠️ Missing required files. Please process content first to generate a summary.")
                # Show which files are missing
                if not files_status['transcript']:
                    st.error("Missing transcript file")
                if not files_status['quotes']:
                    st.error("Missing quotes file")
            else:
                # Load existing summary if available
                existing_summary = load_summary(storage_paths['summary_path']) if files_status['summary'] else None
                
                # Show summary status
                if existing_summary:
                    st.info("✅ Summary already exists")
                else:
                    st.info("ℹ️ No summary generated yet")
                
                # Add generate/regenerate button
                button_text = "🔄 Regenerate Summary" if existing_summary else "✨ Generate Summary"
                if st.button(button_text):
                    try:
                        with st.spinner("Reading files..."):
                            # Read transcript and quotes
                            with open(storage_paths['transcript_path'], 'r', encoding='utf-8') as f:
                                transcript = f.read()
                            quotes = read_quotes(storage_paths['quotes_path'])
                            
                            if not transcript.strip():
                                st.error("❌ Transcript file is empty")
                                st.stop()
                            if not quotes:
                                st.error("❌ No quotes found")
                                st.stop()
                        
                        with st.spinner("Generating summary..."):
                            summary = generate_summary(transcript, quotes, storage_paths['summary_path'])
                            
                            if summary['overview'] != "Error generating summary":
                                # Save the summary
                                save_summary(summary, storage_paths['summary_path'])
                                st.success("✨ Summary generated successfully!")
                                existing_summary = summary  # Update the displayed summary
                                st.rerun()  # Refresh to show new summary
                            else:
                                st.error("Failed to generate summary. Please try again.")
                                logger.error("Summary generation failed: Error response from API")
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
                        logger.exception("Error during summary generation")
                
                # Display existing summary if available
                if existing_summary:
                    with st.container():
                        # Display overview without redundant header
                        st.markdown("### Content Overview")
                        st.write(existing_summary['overview'])
                        
                        # Display key points
                        st.markdown("### Key Points")
                        for point in existing_summary['key_points']:
                            st.markdown(f"• {point}")
                        
                        # Add download button for summary
                        summary_text = f"Content Overview:\n{existing_summary['overview']}\n\nKey Points:\n"
                        summary_text += '\n'.join(f"• {point}" for point in existing_summary['key_points'])
                        
                        st.download_button(
                            "⬇️ Download Summary",
                            summary_text,
                            file_name=f"summary_{st.session_state.current_space_id[:8]}.txt",
                            mime="text/plain"
                        )
        else:
            st.info("Process content in the Main tab to generate a summary.")

    with logs_tab:
        st.markdown("### 🔍 Processing Logs")
        
        # Create an expander for detailed logs with word wrapping
        with st.expander("Detailed Logs", expanded=True):
            try:
                with open('app.log', 'r') as f:
                    logs_content = f.readlines()
                    recent_logs = logs_content[-20:]  # Get last 20 lines
                    with stylable_container(
                        key="detailed_logs",
                        css_styles="""
                            code {
                                white-space: pre-wrap !important;
                            }
                            """
                    ):
                        st.code(''.join(recent_logs), language='text')
            except Exception as e:
                st.warning("No logs available yet")

    with history_tab:
        st.markdown("### 📚 Previously Downloaded Media")
        
        # List all downloaded media
        if DOWNLOADS_DIR.exists():
            media_files = list(DOWNLOADS_DIR.glob("*.mp3"))
            if media_files:
                # Add search/filter box
                search = st.text_input("🔍 Search media files", 
                                     help="Filter by URL or ID")
                
                # Create a radio selection for media files
                media_options = {}
                for media_file in media_files:
                    space_id = media_file.stem
                    original_url = get_url_from_history(space_id)
                    size_mb = media_file.stat().st_size / (1024 * 1024)
                    
                    # Create a formatted label with URL preview
                    url_preview = original_url
                    if len(url_preview) > 60:
                        url_preview = url_preview[:57] + "..."
                    
                    label = f"📁 {space_id}\n💾 {size_mb:.1f} MB\n🔗 {url_preview}"
                    media_options[label] = str(media_file)
                
                # Filter options based on search
                if search:
                    media_options = {k: v for k, v in media_options.items() 
                                   if search.lower() in k.lower() or 
                                   search.lower() in get_url_from_history(Path(v).stem).lower()}
                
                if media_options:
                    selected = st.radio(
                        "Select a media file to manage:",
                        options=list(media_options.keys()),
                        index=None,
                        label_visibility="collapsed"
                    )
                    
                    if selected:
                        selected_path = Path(media_options[selected])
                        space_id = selected_path.stem
                        
                        # Create a nice looking container for file details
                        with st.container():
                            st.markdown("---")
                            st.markdown("#### Selected File Details")
                            
                            # Show full URL in an info box
                            original_url = get_url_from_history(space_id)
                            st.info(f"🔗 Original URL:\n{original_url}")
                            
                            # Show action buttons in a row
                            col1, col2, col3 = st.columns([1, 1, 1])
                            with col1:
                                if st.button("📂 Load", key="load_selected",
                                           help="Load this media in the main view"):
                                    st.session_state.current_space_id = space_id
                                    st.session_state.processing_complete = True
                                    st.rerun()
                            
                            with col2:
                                # Download button
                                with open(selected_path, 'rb') as audio_file:
                                    audio_data = audio_file.read()
                                    st.download_button(
                                        "⬇️ Download",
                                        audio_data,
                                        file_name=selected_path.name,
                                        mime="audio/mpeg",
                                        help="Download the audio file"
                                    )
                            
                            with col3:
                                # Delete button with confirmation
                                if st.button("🗑️ Delete", key="delete_media",
                                           help="Delete all files associated with this media"):
                                    if st.button("⚠️ Confirm Delete", key="confirm_delete"):
                                        try:
                                            # Delete all associated files
                                            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                                            for path in storage_paths.values():
                                                if os.path.exists(str(path)):
                                                    os.remove(str(path))
                                            st.success(f"Deleted all files for {selected_path.stem}")
                                            # Clear session state if the current media was deleted
                                            if space_id == st.session_state.current_space_id:
                                                st.session_state.current_space_id = None
                                                st.session_state.processing_complete = False
                                            # Remove from URL history
                                            history = load_url_history()
                                            if space_id in history:
                                                del history[space_id]
                                                with open(URL_HISTORY_FILE, 'w') as f:
                                                    json.dump(history, f)
                                            time.sleep(1)  # Give time for the success message
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error deleting files: {str(e)}")
                            
                            # Show file status
                            st.markdown("#### File Status")
                            storage_paths = get_storage_paths(str(STORAGE_DIR), space_id)
                            
                            # Check which associated files exist
                            details = {
                                "Audio": os.path.exists(storage_paths['audio_path']),
                                "Transcript": os.path.exists(storage_paths['transcript_path']),
                                "Quotes": os.path.exists(storage_paths['quotes_path']),
                                "Summary": os.path.exists(storage_paths['summary_path'])
                            }
                            
                            # Create status indicators
                            status_cols = st.columns([1, 1, 1, 1])
                            for i, (file_type, exists) in enumerate(details.items()):
                                with status_cols[i]:
                                    icon = "✅" if exists else "❌"
                                    st.markdown(f"**{icon} {file_type}**")
                                    if exists:
                                        # Get size based on file type with literal keys
                                        if file_type == "Audio":
                                            size = Path(storage_paths['audio_path']).stat().st_size / 1024
                                        elif file_type == "Transcript":
                                            size = Path(storage_paths['transcript_path']).stat().st_size / 1024
                                        elif file_type == "Quotes":
                                            size = Path(storage_paths['quotes_path']).stat().st_size / 1024
                                        elif file_type == "Summary":
                                            size = Path(storage_paths['summary_path']).stat().st_size / 1024
                                        else:
                                            size = 0
                else:
                    st.warning("No media files match your search.")
            else:
                st.info("No media files downloaded yet.")
        else:
            st.warning("Downloads directory not found.")

    with help_tab:
        st.subheader("📖 Instructions")
        st.write("""
        1. Paste media URL
        2. Wait for processing (this may take a few minutes)
        3. View and download:
            - Generated quotes
            - Audio recording
            - Full transcript
        """)
        
        st.subheader("💡 Tips")
        st.write("""
        - For best results, use recent media
        - Quotes are automatically formatted for social media
        - You can download everything for offline use
        """)
        
        # Show storage info
        st.subheader("📊 Storage Info")
        total_audio = sum(f.stat().st_size for f in DOWNLOADS_DIR.glob("*.mp3")) / (1024 * 1024)  # MB
        st.write(f"Total audio storage: {total_audio:.1f} MB")
        if total_audio > 1000:  # 1 GB warning
            st.warning("⚠️ Storage is getting full. Consider cleaning old files.") 

else:
    st.stop()  # Do not continue if check_password is not True.
