import yt_dlp
import os

def download_twitter_space(url: str, output_dir: str = "downloads") -> str:
    """
    Download media using yt-dlp.
    
    Args:
        url: URL of the media
        output_dir: Directory to save the downloaded audio
        
    Returns:
        Path to the downloaded audio file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = os.path.join(output_dir, f"{info['id']}.mp3")
            return audio_path
    except Exception as e:
        print(f"Error downloading media: {str(e)}")
        raise
