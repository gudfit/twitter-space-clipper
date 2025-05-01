from typing import Optional, Callable, Dict, Any, List
try:
    import yt_dlp  # type: ignore
except ImportError:
    yt_dlp = None  # type: ignore
import os

def download_twitter_space(url: str, output_path: str, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
    """
    Download media using yt-dlp.
    
    Args:
        url: URL of the media
        output_path: Path where the downloaded audio should be saved
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Path to the downloaded audio file
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove any existing file or directory at output_path
    if os.path.exists(output_path):
        if os.path.isdir(output_path):
            import shutil
            shutil.rmtree(output_path)
        else:
            os.remove(output_path)
    
    # Remove .mp3 extension from output_path for yt-dlp since it will be added by postprocessor
    base_output_path = output_path[:-4] if output_path.endswith('.mp3') else output_path
    
    ydl_opts: Dict[str, Any] = {
        'format': 'bestaudio/best',
        'outtmpl': base_output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'outtmpl_na_placeholder': '',  # Prevent creation of directories for failed downloads
    }
    
    # Add progress hooks if callback provided
    if progress_callback:
        ydl_opts['progress_hooks'] = [progress_callback]
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return output_path
    except Exception as e:
        print(f"Error downloading media: {str(e)}")
        raise
