import streamlit as st
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import json
import hashlib
import shutil
import yt_dlp
import re
from typing import Optional, Dict, Any, List
import time
import logging
from streamlit_extras.stylable_container import stylable_container

# Import core functionality
from core.download import download_twitter_space
from core.transcribe import transcribe_audio
from core.quotes import create_quote_thread
from core.summary import generate_summary, save_summary, load_summary
from core.processor import process_space, get_space_id, get_storage_paths, regenerate_quotes
from utils.api import call_deepseek_api

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
    ⚠️ ffmpeg is not installed or not found in PATH. This is required for processing media.
    
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
if 'selected_media' not in st.session_state:
    st.session_state.selected_media = None
if 'url_history' not in st.session_state:
    st.session_state.url_history = {}
if 'loaded_space_id' not in st.session_state:
    st.session_state.loaded_space_id = None

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
    with open(URL_HISTORY_FILE, 'w') as f:
        json.dump({}, f)

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

class StreamlitProgressCallback:
    """Progress callback that updates Streamlit UI."""
    def __init__(self, container):
        self.container = container
        self.progress_bar = None
        
    def __call__(self, stage: str, progress: float, status: str):
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
        self.progress_bar.progress(progress, f"{emoji} {stage.title()}: {status}")
        
        # Only show completion/error messages outside progress bar
        if stage == "complete":
            self.container.success("✅ Processing complete!")
        elif stage == "error":
            self.container.error(f"❌ Error: {status}")

def process_space_with_ui(url: str, _progress_container) -> dict:
    """Process media URL with Streamlit UI updates."""
    try:
        log_processing_step("Space processing", "started", f"URL: {url}")
        
        # Create progress callback
        progress_callback = StreamlitProgressCallback(_progress_container)
        
        # Process the space
        paths = process_space(url, str(STORAGE_DIR), progress_callback)
        
        if paths:
            st.session_state.processing_complete = True
            return paths
        else:
            st.error("Processing failed. Check the logs for details.")
            return None
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def download_with_progress(url: str, output_dir: str, progress_bar) -> Optional[str]:
    """Download media with progress tracking."""
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
        'quotes_path': str(QUOTES_DIR / f"{space_id}.txt"),
        'summary_path': str(SUMMARIES_DIR / f"{space_id}.json")
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

def chunk_transcript(transcript: str, chunk_size: int = 50000) -> List[str]:
    """Split transcript into manageable chunks while preserving sentence boundaries."""
    chunks = []
    current_chunk = []
    current_size = 0
    
    # Split by sentences (roughly)
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def load_previous_media(space_id: str):
    """Load previously processed media and set up the interface."""
    # Set session state
    st.session_state.current_space_id = space_id
    st.session_state.processing_complete = True
    st.session_state.loaded_space_id = space_id
    
    # Get paths
    paths = get_storage_paths(space_id)
    
    # Verify all files exist
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    return None

def generate_summary(transcript: str, quotes: List[str]) -> Dict[str, str]:
    """Generate a comprehensive summary using both transcript and quotes."""
    from core.summary import generate_summary as core_generate_summary
    return core_generate_summary(transcript, quotes)

def save_summary(summary: Dict[str, Any], path: str) -> None:
    """Save summary to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

def load_summary(path: str) -> Optional[Dict[str, Any]]:
    """Load summary from a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Main app
st.title("🎙️ LinkToQuote")
st.write("Generate quotes and clips from any media url -  no listening required!")

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
        space_id = get_space_id(space_url) if space_url else st.session_state.loaded_space_id
        
        # Save URL to history when processing new media
        if space_url and space_id != st.session_state.current_space_id:
            save_url_history(space_id, space_url)
            st.session_state.processing_complete = False
            st.session_state.current_space_id = space_id
            st.session_state.download_progress = 0
            st.session_state.total_fragments = 0
            st.session_state.current_fragment = 0

        if not st.session_state.processing_complete:
            progress_container = st.container()
            with st.status("Processing media...", expanded=True) as status:
                try:
                    process_result = process_space_with_ui(space_url, progress_container)
                    if process_result:
                        st.session_state.processing_complete = True
                        st.success("✅ Space processed successfully!")
                        
                        # Check if we need to generate summary
                        paths = get_storage_paths(st.session_state.current_space_id)
                        if not os.path.exists(paths['summary_path']):  # Only generate if doesn't exist
                            if os.path.exists(paths['transcript_path']) and os.path.exists(paths['quotes_path']):
                                with st.spinner("Generating summary..."):
                                    # Read transcript and quotes
                                    with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                                        transcript = f.read()
                                    quotes = read_quotes(paths['quotes_path'])
                                    
                                    summary = generate_summary(transcript, quotes)
                                    
                                    if summary['overview'] != "Error generating summary":
                                        # Save the summary
                                        save_summary(summary, paths['summary_path'])
                                        st.success("✨ Summary generated successfully!")
                                    else:
                                        st.error("Failed to generate summary. Please try manually.")
                        else:
                            logger.info("Summary already exists, skipping generation")
                except Exception as e:
                    st.error(f"Error processing Space: {str(e)}")
                    status.update(label="Error!", state="error")
                    st.session_state.processing_complete = False

        # If processing is complete or media was loaded, show results
        if st.session_state.processing_complete:
            process_result = get_storage_paths(space_id)
            
            # Show original URL for loaded content
            if st.session_state.loaded_space_id:
                original_url = get_url_from_history(space_id)
                st.info(f"🔗 Loaded content from: {original_url}")

            # Display tabs for different outputs
            content_tab1, content_tab2, content_tab3 = st.tabs(["📝 Quotes", "🎵 Audio", "📄 Transcript"])
            
            with content_tab1:
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
            
            with content_tab2:
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
            
            with content_tab3:
                st.subheader("Transcript")
                if os.path.exists(process_result['transcript_path']):
                    with open(process_result['transcript_path'], 'r') as f:
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
        paths = get_storage_paths(st.session_state.current_space_id)
        
        # Check for required files first
        files_status = {
            'transcript': os.path.exists(paths['transcript_path']),
            'quotes': os.path.exists(paths['quotes_path']),
            'summary': os.path.exists(paths['summary_path'])
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
            existing_summary = load_summary(paths['summary_path']) if files_status['summary'] else None
            
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
                        with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                            transcript = f.read()
                        quotes = read_quotes(paths['quotes_path'])
                        
                        if not transcript.strip():
                            st.error("❌ Transcript file is empty")
                            st.stop()
                        if not quotes:
                            st.error("❌ No quotes found")
                            st.stop()
                    
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(transcript, quotes)
                        
                        if summary['overview'] != "Error generating summary":
                            # Save the summary
                            save_summary(summary, paths['summary_path'])
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
                            with open(selected_path, 'rb') as f:
                                st.download_button(
                                    "⬇️ Download",
                                    f,
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
                                        paths = get_storage_paths(space_id)
                                        for path in paths.values():
                                            if os.path.exists(path):
                                                os.remove(path)
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
                        paths = get_storage_paths(space_id)
                        
                        # Check which associated files exist
                        details = {
                            "Audio": os.path.exists(paths['audio_path']),
                            "Transcript": os.path.exists(paths['transcript_path']),
                            "Quotes": os.path.exists(paths['quotes_path']),
                            "Summary": os.path.exists(paths['summary_path'])
                        }
                        
                        # Create status indicators
                        status_cols = st.columns(4)
                        for i, (file_type, exists) in enumerate(details.items()):
                            with status_cols[i]:
                                icon = "✅" if exists else "❌"
                                st.markdown(f"**{icon} {file_type}**")
                                if exists:
                                    size = Path(paths[file_type.lower() + '_path']).stat().st_size / 1024
                                    if size < 1024:
                                        st.caption(f"{size:.1f} KB")
                                    else:
                                        st.caption(f"{size/1024:.1f} MB")
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
