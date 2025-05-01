from typing import Optional, Callable, Dict, Any, List
try:
    import yt_dlp
except ImportError:
    yt_dlp = None
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
    # First check if the MP3 already exists and is valid
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        if progress_callback:
            progress_callback({
                'status': 'finished',
                'filename': output_path,
                'downloaded_bytes': os.path.getsize(output_path),
                'total_bytes': os.path.getsize(output_path)
            })
        return output_path
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
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
