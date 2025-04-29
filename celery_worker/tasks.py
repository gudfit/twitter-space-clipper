"""Core Celery tasks for processing media files."""
import os
import json
import logging
from typing import Dict, List, Optional

from celery_worker.celery_app import celery_app
from core.transcribe import transcribe_audio
from core.quotes import create_quote_thread
from core.summary import generate_summary

# Configure module logger
logger = logging.getLogger(__name__)


@celery_app.task
def add(x, y):
    return x + y

@celery_app.task(bind=True)
def transcribe_task(self, audio_path: str, transcript_path: str) -> Optional[str]:
    """Transcribe audio file task.
    
    Args:
        audio_path: Path to the audio file
        transcript_path: Path to save the transcript
        
    Returns:
        The transcript text if successful, None otherwise
    """
    try:
        logger.info(f"Starting transcription of {audio_path}")
        transcript = transcribe_audio(audio_path)
        
        if transcript:
            # Save transcript
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            logger.info(f"Transcription saved to {transcript_path}")
            return transcript
        else:
            logger.error("Transcription failed - no transcript generated")
            return None
            
    except Exception as e:
        logger.exception("Error during transcription")
        raise

@celery_app.task(bind=True)
def generate_quotes_task(self, transcript: str, space_info: Dict, quotes_path: str) -> Optional[List[str]]:
    """Generate quotes from transcript task.
    
    Args:
        transcript: The transcript text
        space_info: Dictionary containing space metadata
        quotes_path: Path to save the quotes
        
    Returns:
        List of generated quotes if successful, None otherwise
    """
    try:
        logger.info("Starting quote generation")
        quotes = create_quote_thread(transcript, space_info)
        
        if quotes:
            # Save quotes
            with open(quotes_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(quotes))
            logger.info(f"Quotes saved to {quotes_path}")
            return quotes
        else:
            logger.error("Quote generation failed - no quotes generated")
            return None
            
    except Exception as e:
        logger.exception("Error during quote generation")
        raise

@celery_app.task(bind=True)
def generate_summary_task(self, transcript: str, quotes: List[str], summary_path: str) -> Optional[Dict]:
    """Generate summary from transcript and quotes task.
    
    Args:
        transcript: The transcript text
        quotes: List of generated quotes
        summary_path: Path to save the summary
        
    Returns:
        Summary dictionary if successful, None otherwise
    """
    try:
        logger.info("Starting summary generation")
        summary = generate_summary(transcript, quotes)
        
        if summary and summary['overview'] != "Error generating summary":
            # Save summary
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Summary saved to {summary_path}")
            return summary
        else:
            logger.error("Summary generation failed")
            return None
            
    except Exception as e:
        logger.exception("Error during summary generation")
        raise