"""Core functionality for processing media files."""
import os
import shutil
import logging
import json
from typing import Dict, Optional, Protocol, Any, Callable, List
from pathlib import Path

from core.download import download_twitter_space
from core.transcribe import transcribe_audio
from core.quotes import create_quote_thread
from core.summary import generate_summary
from utils.file_utils import clean_filename

# Configure module logger
logger = logging.getLogger(__name__)

class ProgressCallback(Protocol):
    """Protocol for progress tracking callbacks."""
    def __call__(self, stage: str, progress: float, status: str) -> None: ...

def get_space_id(url: str) -> str:
    """Extract space ID from URL and create a hash for storage.
    
    Args:
        url: The URL to process
        
    Returns:
        A hash string identifying the space
    """
    import hashlib
    space_id = url.strip('/').split('/')[-1]
    return hashlib.md5(space_id.encode()).hexdigest()

def get_storage_paths(storage_root: str, space_id: str) -> Dict[str, str]:
    """Get paths for storing space data.
    
    Args:
        storage_root: Root directory for storage
        space_id: Unique identifier for the space
        
    Returns:
        Dictionary of paths for different file types
    """
    storage_dir = Path(storage_root)
    downloads_dir = storage_dir / "downloads"
    transcripts_dir = storage_dir / "transcripts"
    quotes_dir = storage_dir / "quotes"
    summaries_dir = storage_dir / "summaries"
    
    # Ensure directories exist
    for dir_path in [downloads_dir, transcripts_dir, quotes_dir, summaries_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'audio_path': str(downloads_dir / f"{space_id}.mp3"),
        'transcript_path': str(transcripts_dir / f"{space_id}.txt"),
        'quotes_path': str(quotes_dir / f"{space_id}.txt"),
        'summary_path': str(summaries_dir / f"{space_id}.json")
    }

def process_space(
    url: str,
    storage_root: str,
    progress_callback: Optional[ProgressCallback] = None,
    download_callback: Optional[Callable] = None
) -> Optional[Dict[str, str]]:
    """Process media URL and return paths to generated files.
    
    Args:
        url: URL of the media to process
        storage_root: Root directory for storing files
        progress_callback: Optional callback for progress updates
        download_callback: Optional callback for download progress
        
    Returns:
        Dictionary of paths to generated files or None on error
        
    Example:
        >>> def progress(stage, progress, status):
        ...     print(f"{stage}: {progress*100:.0f}% - {status}")
        >>> paths = process_space("https://example.com/media", "storage", progress)
    """
    try:
        logger.info(f"Starting space processing for URL: {url}")
        space_id = get_space_id(url)
        paths = get_storage_paths(storage_root, space_id)
        
        # Check if all files already exist
        if all(os.path.exists(p) for p in paths.values()):
            logger.info("All files already exist, skipping processing")
            if progress_callback:
                progress_callback("complete", 1.0, "Files already exist")
            return paths

        # Download audio if needed
        if not os.path.exists(paths['audio_path']):
            logger.info("Starting audio download")
            if progress_callback:
                progress_callback("download", 0.0, "Starting download...")
                
            downloaded_file = download_twitter_space(url, os.path.dirname(paths['audio_path']))
            
            if not downloaded_file:
                logger.error("Failed to download audio")
                raise Exception("Failed to download audio")
                
            if not downloaded_file.endswith('.mp3'):
                logger.warning("Downloaded file is not MP3")
                paths['audio_path'] = downloaded_file
            else:
                shutil.move(downloaded_file, paths['audio_path'])
                
            if progress_callback:
                progress_callback("download", 1.0, "Download complete")

        # Only continue with transcription if we have an MP3 file
        if not paths['audio_path'].endswith('.mp3'):
            logger.error("Audio file is not in MP3 format")
            raise Exception("Audio file must be in MP3 format")

        # Transcribe audio if needed
        if not os.path.exists(paths['transcript_path']):
            logger.info("Starting transcription")
            if progress_callback:
                progress_callback("transcribe", 0.0, "Starting transcription...")
                
            transcript = transcribe_audio(paths['audio_path'])
            if not transcript:
                logger.error("Failed to transcribe audio")
                raise Exception("Failed to transcribe audio")
                
            with open(paths['transcript_path'], "w", encoding="utf-8") as f:
                f.write(transcript)
                
            if progress_callback:
                progress_callback("transcribe", 1.0, "Transcription complete")

        # Generate quotes if needed
        if not os.path.exists(paths['quotes_path']):
            logger.info("Starting quote generation")
            if progress_callback:
                progress_callback("quotes", 0.0, "Generating quotes...")
                
            with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                transcript = f.read()
                
            quotes = create_quote_thread(transcript, {"url": url})
            if not quotes:
                logger.error("Failed to generate quotes")
                raise Exception("Failed to generate quotes")
                
            with open(paths['quotes_path'], "w", encoding="utf-8") as f:
                f.write("\n\n".join(quotes) if isinstance(quotes, list) else quotes)
                
            if progress_callback:
                progress_callback("quotes", 1.0, "Quote generation complete")

        # Generate summary if needed
        if not os.path.exists(paths['summary_path']):
            logger.info("Starting summary generation")
            if progress_callback:
                progress_callback("summary", 0.0, "Generating summary...")
                
            with open(paths['transcript_path'], 'r', encoding='utf-8') as f:
                transcript = f.read()
                
            # Read quotes
            with open(paths['quotes_path'], 'r', encoding='utf-8') as f:
                quotes_text = f.read()
                quotes = [q.strip() for q in quotes_text.split('\n\n') if q.strip()]
                
            summary = generate_summary(transcript, quotes)
            if summary['overview'] != "Error generating summary":
                with open(paths['summary_path'], 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2)
                    
                if progress_callback:
                    progress_callback("summary", 1.0, "Summary generation complete")
            else:
                logger.error("Failed to generate summary")
                if progress_callback:
                    progress_callback("summary", 1.0, "Summary generation failed")

        logger.info("Space processing completed successfully")
        if progress_callback:
            progress_callback("complete", 1.0, "Processing complete")
            
        return paths
        
    except Exception as e:
        logger.exception("Error processing space")
        if progress_callback:
            progress_callback("error", 0.0, str(e))
        return None

def regenerate_quotes(transcript_path: str, quotes_path: str, url: str) -> List[str]:
    """Regenerate quotes from transcript.
    
    Args:
        transcript_path: Path to the transcript file
        quotes_path: Path to save the generated quotes
        url: Original URL of the media
        
    Returns:
        List of generated quotes
    """
    logger.info(f"Regenerating quotes for {url}")
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Add space info to match initial generation
        space_info = {"url": url}
        
        quotes = create_quote_thread(transcript, space_info)
        if not quotes:
            logger.warning("No quotes were generated")
            return []
            
        # Save new quotes
        with open(quotes_path, 'w', encoding='utf-8') as f:
            if isinstance(quotes, list):
                f.write("\n\n".join(quotes))
            else:
                f.write(quotes)
        
        logger.info(f"Successfully regenerated {len(quotes) if isinstance(quotes, list) else 1} quotes")
        return quotes if isinstance(quotes, list) else quotes.split('\n\n')
        
    except Exception as e:
        logger.exception("Error regenerating quotes")
        return [] 