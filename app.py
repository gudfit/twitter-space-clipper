import streamlit as st
import os
from pathlib import Path
import sys
import tempfile
import subprocess
from xdownload_space import download_twitter_space
from transcribe import transcribe_audio
from xquotes import create_quote_thread
import json
import hashlib
import shutil
import yt_dlp
import re
from typing import Optional, Dict, Any, List
import time
import logging
from streamlit_extras.stylable_container import stylable_container

# Configure logging
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

def log_processing_step(step: str, status: str = "started", details: str = None):
    """Helper function to log processing steps consistently"""
    message = f"{step} {status}"
    if details:
        message += f": {details}"
    logger.info(message)

# Disable watchdog for PyTorch modules to prevent custom class errors
import streamlit.watcher.path_watcher
original_watch_file = streamlit.watcher.path_watcher.watch_file

def patched_watch_file(filepath, *args, **kwargs):
    if 'torch' in filepath or '_C' in filepath:
        return None
    return original_watch_file(filepath, *args, **kwargs)

streamlit.watcher.path_watcher.watch_file = patched_watch_file

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Configure Streamlit page
st.set_page_config(
    page_title="LinkToQuotes",
    page_icon="🎙️",
    layout="wide"
)

# Check for ffmpeg at startup
if not check_ffmpeg():
    st.error("""
    ⚠️ ffmpeg is not installed or not found in PATH. This is required for processing Twitter Spaces.
    
    To install ffmpeg:
    - Ubuntu/Debian: `sudo apt-get install ffmpeg`
    - macOS: `brew install ffmpeg`
    - Windows: Download from https://ffmpeg.org/download.html
    
    After installing, please restart the application.
    """)
    st.stop()

# Initialize session state for progress tracking
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'current_space_id' not in st.session_state:
    st.session_state.current_space_id = None
if 'download_progress' not in st.session_state:
    st.session_state.download_progress = 0
if 'total_fragments' not in st.session_state:
    st.session_state.total_fragments = 0
if 'current_fragment' not in st.session_state:
    st.session_state.current_fragment = 0
if 'regenerating_quotes' not in st.session_state:
    st.session_state.regenerating_quotes = False

# Create persistent storage directories
STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR = STORAGE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR = STORAGE_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
QUOTES_DIR = STORAGE_DIR / "quotes"
QUOTES_DIR.mkdir(exist_ok=True)

class ProgressCallback:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar
        self.total_fragments = 0
        self.downloaded_fragments = 0

    def format_bytes(self, bytes_num: float) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_num < 1024:
                return f"{bytes_num:.1f} {unit}"
            bytes_num /= 1024
        return f"{bytes_num:.1f} TB"

    def format_eta(self, seconds: float) -> str:
        """Format ETA seconds to human readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"

    def __call__(self, d: Dict[str, Any]) -> None:
        if d['status'] == 'downloading':
            # Extract total fragments if not already set
            if not self.total_fragments and 'total_fragments' in d:
                self.total_fragments = d['total_fragments']
                st.session_state.total_fragments = self.total_fragments

            # Update progress based on fragment number
            if 'fragment_index' in d:
                self.downloaded_fragments = d['fragment_index'] + 1
                st.session_state.current_fragment = self.downloaded_fragments
                if self.total_fragments:
                    progress = self.downloaded_fragments / self.total_fragments
                    status_text = f"Downloading... ({self.downloaded_fragments}/{self.total_fragments} fragments)"
                    
                    # Add speed and ETA if available
                    if 'speed' in d and d['speed'] is not None:
                        status_text += f" | {self.format_bytes(d['speed'])}/s"
                    if 'eta' in d and d['eta'] is not None:
                        status_text += f" | ETA: {self.format_eta(d['eta'])}"
                    
                    self.progress_bar.progress(progress, status_text)

            # Update progress based on downloaded bytes
            elif 'downloaded_bytes' in d and 'total_bytes_estimate' in d:
                progress = d['downloaded_bytes'] / d['total_bytes_estimate']
                status_text = f"Downloading... {d['_percent_str']}"
                
                # Add speed and ETA if available
                if 'speed' in d and d['speed'] is not None:
                    status_text += f" | {self.format_bytes(d['speed'])}/s"
                if 'eta' in d and d['eta'] is not None:
                    status_text += f" | ETA: {self.format_eta(d['eta'])}"
                
                self.progress_bar.progress(progress, status_text)

def download_with_progress(url: str, output_dir: str, progress_bar) -> Optional[str]:
    """Download Twitter Space with progress tracking."""
    # First download without post-processing
    download_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'progress_hooks': [ProgressCallback(progress_bar)],
        'keepvideo': True,  # Keep the original file
        'postprocessors': [],  # No post-processing yet
    }
    
    try:
        # First, just download the file
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            original_file = os.path.join(output_dir, f"{info['id']}.{info['ext']}")
            
            if not os.path.exists(original_file):
                st.error("Download failed - file not found")
                return None
                
            # Show the raw file location
            st.info(f"Raw file downloaded to: {original_file}")
            
            # Now try to convert to MP3 if ffmpeg is available
            if check_ffmpeg():
                progress_bar.progress(0.95, "Converting to MP3...")
                mp3_file = os.path.join(output_dir, f"{info['id']}.mp3")
                
                try:
                    # Convert to MP3 using ffmpeg
                    subprocess.run([
                        'ffmpeg', '-y', '-i', original_file,
                        '-vn', '-acodec', 'libmp3lame',
                        mp3_file
                    ], check=True, capture_output=True)
                    
                    # Cleanup original file after successful conversion
                    os.remove(original_file)
                    progress_bar.progress(1.0, "Conversion complete!")
                    return mp3_file
                    
                except subprocess.SubprocessError as e:
                    st.error(f"MP3 conversion failed: {str(e)}")
                    st.info("You can still find the raw file at: " + original_file)
                    return original_file
            else:
                st.warning("""
                ℹ️ ffmpeg not found - file downloaded but not converted to MP3.
                Please install ffmpeg to enable audio conversion:
                
                ```bash
                sudo apt-get update
                sudo apt-get install ffmpeg
                ```
                
                The raw file is available at: """ + original_file)
                return original_file
                
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return None

def get_space_id(url: str) -> str:
    """Extract space ID from URL and create a hash for storage."""
    space_id = url.strip('/').split('/')[-1]
    return hashlib.md5(space_id.encode()).hexdigest()

def get_storage_paths(space_id: str) -> dict:
    """Get paths for storing space data."""
    return {
        'audio_path': str(DOWNLOADS_DIR / f"{space_id}.mp3"),
        'transcript_path': str(TRANSCRIPTS_DIR / f"{space_id}.txt"),
        'quotes_path': str(QUOTES_DIR / f"{space_id}.txt")
    }

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

@st.cache_data(show_spinner=False)
def process_space(url: str, _progress_container) -> dict:
    """Process a Twitter Space URL and return paths to generated files."""
    try:
        log_processing_step("Space processing", "started", f"URL: {url}")
        space_id = get_space_id(url)
        paths = get_storage_paths(space_id)
        
        if all(os.path.exists(p) for p in paths.values()):
            log_processing_step("Space processing", "skipped", "Files already exist")
            st.session_state.processing_complete = True  # Set processing complete since files exist
            return paths

        # Download progress bar
        if not os.path.exists(paths['audio_path']):
            with _progress_container:
                log_processing_step("Audio download", "started")
                st.write("⬇️ Downloading Space recording...")
                download_progress = st.progress(0, "Preparing download...")
                downloaded_file = download_with_progress(url, str(DOWNLOADS_DIR), download_progress)
                
                if not downloaded_file:
                    log_processing_step("Audio download", "failed")
                    raise Exception("Failed to download Space")
                
                log_processing_step("Audio download", "completed", f"File: {downloaded_file}")
                
                # If the file isn't already MP3, we'll need to convert it later
                if not downloaded_file.endswith('.mp3'):
                    st.warning("File downloaded but needs conversion to MP3. Install ffmpeg and try again.")
                    paths['audio_path'] = downloaded_file  # Store the temporary path
                else:
                    shutil.move(downloaded_file, paths['audio_path'])
                    download_progress.progress(1.0, "Download and conversion complete!")

        # Only continue with transcription if we have an MP3 file
        if not paths['audio_path'].endswith('.mp3'):
            st.error("Please install ffmpeg and restart to continue processing")
            return None

        # Transcription progress
        if not os.path.exists(paths['transcript_path']):
            with _progress_container:
                log_processing_step("Transcription", "started")
                st.write("🎯 Transcribing audio...")
                transcribe_progress = st.progress(0, "Starting transcription...")
                transcript = transcribe_audio(paths['audio_path'])
                if not transcript:
                    log_processing_step("Transcription", "failed", "Failed to transcribe audio")
                    raise Exception("Failed to transcribe audio")
                with open(paths['transcript_path'], "w", encoding="utf-8") as f:
                    f.write(transcript)
                transcribe_progress.progress(1.0, "Transcription complete!")

        # Quote generation progress
        if not os.path.exists(paths['quotes_path']):
            with _progress_container:
                log_processing_step("Quote generation", "started")
                st.write("✍️ Generating quotes...")
                quote_progress = st.progress(0, "Processing transcript...")
                with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                    transcript = f.read()
                quotes = create_quote_thread(transcript, {"url": url})
                with open(paths['quotes_path'], "w", encoding="utf-8") as f:
                    f.write("\n\n".join(quotes) if isinstance(quotes, list) else quotes)
                quote_progress.progress(1.0, "Quote generation complete!")

        return paths
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def regenerate_quotes(transcript_path: str, quotes_path: str, url: str) -> List[str]:
    """Regenerate quotes from transcript."""
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Add space info to match initial generation
        space_info = {"url": url}
        
        # Call with error handling
        try:
            quotes = create_quote_thread(transcript, space_info)
        except Exception as e:
            st.error(f"Error calling quote generation API: {str(e)}")
            return []
        
        if not quotes:
            st.warning("No quotes were generated. Please try again.")
            return []
            
        # Save new quotes
        with open(quotes_path, 'w', encoding='utf-8') as f:
            if isinstance(quotes, list):
                f.write("\n\n".join(quotes))
            else:
                f.write(quotes)
        
        return quotes if isinstance(quotes, list) else quotes.split('\n\n')
    except Exception as e:
        st.error(f"Error regenerating quotes: {str(e)}")
        return []

def read_quotes(file_path: str) -> List[str]:
    """Read and parse quotes from a file.

    Args:
        file_path (str): Path to the quotes file.

    Returns:
        List[str]: List of cleaned quotes, with headers and instructions removed.

    Raises:
        FileNotFoundError: If the quotes file doesn't exist.
        IOError: If there are issues reading the file.
    """
    logger.info(f"Reading quotes from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            quotes_text = f.read()
            # Split by double newlines and clean each quote
            raw_quotes = [q.strip() for q in quotes_text.split('\n\n') if q.strip()]
            
            # Filter out headers and instructions, extract actual quotes
            quotes = []
            for quote in raw_quotes:
                # Skip pure instruction lines
                if quote.startswith(('Here are', 'Each quote')):
                    continue
                    
                # If it's a numbered quote, extract the actual quote text
                if re.match(r'^\d+\.\s+', quote):
                    # Extract text between ** markers if present
                    match = re.search(r'\*\*(.*?)\*\*', quote)
                    if match:
                        quote = match.group(1)
                    else:
                        # If no ** markers, take everything after the number
                        quote = re.sub(r'^\d+\.\s+', '', quote)
                
                # Clean up the quote
                quote = quote.strip().strip('"')  # Remove quotes and whitespace
                if quote:  # Only add non-empty quotes
                    quotes.append(quote)
            
            logger.info(f"Successfully read {len(quotes)} quotes")
            return quotes
    except FileNotFoundError as e:
        logger.error(f"Quotes file not found: {str(e)}")
        raise
    except IOError as e:
        logger.error(f"Error reading quotes file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error reading quotes: {str(e)}")
        return []

def display_quotes(quotes: List[str], container: st.container) -> None:
    """Display quotes in the Streamlit interface with copy buttons.

    Args:
        quotes (List[str]): List of quotes to display.
        container (st.container): Streamlit container to render quotes in.

    Note:
        Uses st.code for built-in copy functionality with custom CSS for word wrapping.
    """
    logger.info(f"Starting quote display for {len(quotes)} quotes")
    
    with stylable_container(
        "codeblock",
        """
        code {
            white-space: pre-wrap !important;
        }
        """,
    ):
        # Show quote stats
        st.info(f"📊 Generated {len(quotes)} quotes")
        
        # Display each quote
        for i, quote in enumerate(quotes, 1):
            if not quote.strip():
                logger.debug(f"Skipping empty quote at index {i}")
                continue
                
            logger.debug(f"Displaying quote {i}: {quote[:50]}...")
            
            # Display quote with copy button using st.code
            st.code(quote, language=None)  # None for plain text formatting
            
            # Add a small visual separator between quotes
            st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2'>", unsafe_allow_html=True)
        
        # Download button for all quotes
        if quotes:
            st.download_button(
                "⬇️ Download All Quotes",
                '\n\n'.join(quotes),
                file_name=f"quotes_{st.session_state.current_space_id[:8]}.txt",
                mime="text/plain"
            )

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main app
st.title("🎙️ Twitter Space Content Generator")
st.write("Generate quotes and clips from Twitter Spaces - no listening required!")

# Create two columns - main content and logs
main_col, log_col = st.columns([2, 1])

with main_col:
    # URL input
    space_url = st.text_input("Paste Twitter Space URL:", placeholder="https://twitter.com/i/spaces/...")

    if space_url:
        space_id = get_space_id(space_url)
        
        if space_id != st.session_state.current_space_id:
            st.session_state.processing_complete = False
            st.session_state.current_space_id = space_id
            st.session_state.download_progress = 0
            st.session_state.total_fragments = 0
            st.session_state.current_fragment = 0

        if not st.session_state.processing_complete:
            progress_container = st.container()
            with st.status("Processing Twitter Space...", expanded=True) as status:
                try:
                    process_result = process_space(space_url, progress_container)
                    if process_result:
                        st.session_state.processing_complete = True
                        st.success("✅ Space processed successfully!")
                except Exception as e:
                    st.error(f"Error processing Space: {str(e)}")
                    status.update(label="Error!", state="error")
                    st.session_state.processing_complete = False

        # If processing is complete, show results
        if st.session_state.processing_complete:
            process_result = get_storage_paths(space_id)
            
            # Display tabs for different outputs
            tab1, tab2, tab3 = st.tabs(["📝 Quotes", "🎵 Audio", "📄 Transcript"])
            
            with tab1:
                st.subheader("Generated Quotes")
                if os.path.exists(process_result['quotes_path']):
                    # Add regenerate button at the top
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        if st.button("🔄 Regenerate", key="regenerate_quotes"):
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
                                                process_result['transcript_path'],
                                                process_result['quotes_path'],
                                                space_url
                                            )
                                            
                                            # Get the logs
                                            logs = mystdout.getvalue()
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
                    quotes = read_quotes(process_result['quotes_path'])
                    if quotes:  # Only try to display if we have quotes
                        display_quotes(quotes, st.container())
                    else:
                        st.warning("No quotes found in the file. Try regenerating the quotes.")
            
            with tab2:
                st.subheader("Audio")
                if os.path.exists(process_result['audio_path']):
                    audio_size = os.path.getsize(process_result['audio_path']) / (1024 * 1024)  # MB
                    st.write(f"Audio file size: {audio_size:.1f} MB")
                    
                    with open(process_result['audio_path'], 'rb') as f:
                        st.download_button(
                            "⬇️ Download Full Recording",
                            f,
                            file_name=f"space_{space_id[:8]}.mp3",
                            mime="audio/mpeg"
                        )
            
            with tab3:
                st.subheader("Transcript")
                if os.path.exists(process_result['transcript_path']):
                    with open(process_result['transcript_path'], 'r') as f:
                        transcript = f.read()
                    st.text_area("Full Transcript", transcript, height=300)
                    st.download_button(
                        "⬇️ Download Transcript",
                        transcript,
                        file_name=f"transcript_{space_id[:8]}.txt",
                        mime="text/plain"
                    )

with log_col:
    st.markdown("### 🔍 Processing Logs")
    
    # Create an expander for detailed logs with word wrapping
    with st.expander("Detailed Logs", expanded=True):
        try:
            with open('app.log', 'r') as f:
                logs = f.readlines()
                recent_logs = logs[-20:]  # Get last 20 lines
                # Apply word wrapping using CSS
                st.markdown(
                    """
                    <style>
                        .stMarkdown pre {
                            white-space: pre-wrap !important;
                            word-wrap: break-word !important;
                        }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.code(''.join(recent_logs), language='text')
        except Exception as e:
            st.warning("No logs available yet")
    

# Sidebar with instructions and storage info
with st.sidebar:
    st.subheader("📖 Instructions")
    st.write("""
    1. Paste a Twitter Space URL
    2. Wait for processing (this may take a few minutes)
    3. View and download:
        - Generated quotes
        - Audio recording
        - Full transcript
    """)
    
    st.subheader("💡 Tips")
    st.write("""
    - For best results, use recent Twitter Spaces
    - Quotes are automatically formatted for social media
    - You can download everything for offline use
    """)
    
    # Show storage info
    st.subheader("📊 Storage Info")
    total_audio = sum(f.stat().st_size for f in DOWNLOADS_DIR.glob("*.mp3")) / (1024 * 1024)  # MB
    st.write(f"Total audio storage: {total_audio:.1f} MB")
    if total_audio > 1000:  # 1 GB warning
        st.warning("⚠️ Storage is getting full. Consider cleaning old files.") 