"""Core Celery tasks for processing media files."""
import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from celery import Celery, chain, group
from pathlib import Path

from celery_worker.celery_app import celery_app

# Configure module logger
logger = logging.getLogger(__name__)

# Initialize Celery
app = celery_app

# Configure task routing
app.conf.task_routes = {
    'celery_worker.tasks.download_media': {'queue': 'download'},
    'celery_worker.tasks.transcribe_media': {'queue': 'transcribe'},
    'celery_worker.tasks.generate_quotes_task': {'queue': 'generate'},
    'celery_worker.tasks.generate_summary_task': {'queue': 'generate'},
}

def update_process_state(storage_dir: str, space_id: str, 
                        stage: str, progress: float, 
                        status: str = "", files: Optional[Dict[str, bool]] = None) -> None:
    """Update the process state file with current progress."""
    # Import here to avoid circular imports
    from core.processor import save_process_state
    
    state = {
        'status': 'processing',
        'stage': stage,
        'progress': progress,
        'stage_status': status,
        'last_updated': datetime.now().isoformat(),
    }
    if files:
        state['files'] = files
    
    save_process_state(storage_dir, space_id, state)

@app.task(bind=True, queue='download')
def download_media(self, url: str, storage_dir: str) -> Dict[str, Any]:
    """Download media from URL."""
    logger.info(f"Starting download task for URL: {url}")
    # Import here to avoid circular imports
    from core.processor import get_storage_paths
    from core.download import download_twitter_space
    
    space_id = url.strip('/').split('/')[-1]
    storage_paths = get_storage_paths(storage_dir, space_id)
    
    try:
        # Update state to downloading
        update_process_state(storage_dir, space_id, 'download', 0.0)
        
        # Download the media
        download_path = download_twitter_space(url, storage_paths['audio_path'])
        
        # Update state with completion
        files = {'audio': True, 'transcript': False, 'quotes': False, 'summary': False}
        update_process_state(storage_dir, space_id, 'download', 1.0, 
                           "Download complete", files)
        
        logger.info(f"Download complete for URL: {url}")
        return {
            'space_id': space_id,
            'storage_dir': storage_dir,
            'audio_path': download_path,
            'success': True
        }
    except Exception as e:
        logger.error(f"Download failed for URL {url}: {str(e)}")
        # Update state with error
        state = {
            'status': 'error',
            'error': str(e),
            'stage': 'download',
            'last_updated': datetime.now().isoformat()
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, state)
        raise

@app.task(bind=True, queue='transcribe')
def transcribe_media(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe downloaded media."""
    logger.info(f"Starting transcription task for space_id: {task_result.get('space_id')}")
    # Import here to avoid circular imports
    from core.processor import get_storage_paths
    from core.transcribe import transcribe_audio
    
    space_id = task_result['space_id']
    storage_dir = task_result['storage_dir']
    audio_path = task_result['audio_path']
    storage_paths = get_storage_paths(storage_dir, space_id)
    
    try:
        # Update state to transcribing
        update_process_state(storage_dir, space_id, 'transcribe', 0.0)
        
        # Transcribe the audio
        transcript = transcribe_audio(audio_path)
        
        # Save transcript
        os.makedirs(os.path.dirname(storage_paths['transcript_path']), exist_ok=True)
        with open(storage_paths['transcript_path'], 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Update state with completion
        files = {'audio': True, 'transcript': True, 'quotes': False, 'summary': False}
        update_process_state(storage_dir, space_id, 'transcribe', 1.0, 
                           "Transcription complete", files)
        
        logger.info(f"Transcription complete for space_id: {space_id}")
        task_result.update({
            'transcript_path': storage_paths['transcript_path'],
            'transcript': transcript
        })
        return task_result
    except Exception as e:
        logger.error(f"Transcription failed for space_id {space_id}: {str(e)}")
        # Update state with error
        state = {
            'status': 'error',
            'error': str(e),
            'stage': 'transcribe',
            'last_updated': datetime.now().isoformat()
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, state)
        raise

@app.task(bind=True, queue='generate')
def generate_quotes_task(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate quotes from transcript."""
    logger.info(f"Starting quote generation for space_id: {task_result.get('space_id')}")
    # Import here to avoid circular imports
    from core.processor import get_storage_paths
    from core.quotes import create_quote_thread
    
    space_id = task_result['space_id']
    storage_dir = task_result['storage_dir']
    storage_paths = get_storage_paths(storage_dir, space_id)
    
    try:
        # Update state to generating quotes
        update_process_state(storage_dir, space_id, 'quotes', 0.0)
        
        # Generate quotes using create_quote_thread
        quotes = create_quote_thread(task_result['transcript'], storage_paths['quotes_path'])
        
        # Update state with completion
        files = {'audio': True, 'transcript': True, 'quotes': True, 'summary': False}
        update_process_state(storage_dir, space_id, 'quotes', 1.0, 
                           "Quote generation complete", files)
        
        logger.info(f"Quote generation complete for space_id: {space_id}")
        task_result.update({
            'quotes_path': storage_paths['quotes_path'],
            'quotes': quotes
        })
        return task_result
    except Exception as e:
        logger.error(f"Quote generation failed for space_id {space_id}: {str(e)}")
        # Update state with error
        state = {
            'status': 'error',
            'error': str(e),
            'stage': 'quotes',
            'last_updated': datetime.now().isoformat()
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, state)
        raise

@app.task(bind=True, queue='generate')
def generate_summary_task(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary from transcript and quotes."""
    logger.info(f"Starting summary generation for space_id: {task_result.get('space_id')}")
    # Import here to avoid circular imports
    from core.processor import get_storage_paths
    from core.summary import generate_summary
    
    space_id = task_result['space_id']
    storage_dir = task_result['storage_dir']
    storage_paths = get_storage_paths(storage_dir, space_id)
    
    try:
        # Update state to generating summary
        update_process_state(storage_dir, space_id, 'summary', 0.0)
        
        # Generate summary
        summary = generate_summary(task_result['transcript'], 
                                 task_result['quotes'], 
                                 storage_paths['summary_path'])
        
        # Update state with completion
        files = {'audio': True, 'transcript': True, 'quotes': True, 'summary': True}
        state = {
            'status': 'complete',
            'stage': 'summary',
            'progress': 1.0,
            'stage_status': "Processing complete",
            'last_updated': datetime.now().isoformat(),
            'files': files
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, state)
        
        logger.info(f"Summary generation complete for space_id: {space_id}")
        task_result.update({
            'summary_path': storage_paths['summary_path'],
            'summary': summary
        })
        return task_result
    except Exception as e:
        logger.error(f"Summary generation failed for space_id {space_id}: {str(e)}")
        # Update state with error
        state = {
            'status': 'error',
            'error': str(e),
            'stage': 'summary',
            'last_updated': datetime.now().isoformat()
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, state)
        raise

def process_space_chain(url: str, storage_dir: str) -> chain:
    """Create a chain of tasks to process a space."""
    logger.info(f"Creating task chain for URL: {url}")
    task_chain = chain(
        download_media.s(url, storage_dir),
        transcribe_media.s(),
        generate_quotes_task.s(),
        generate_summary_task.s()
    )
    logger.info(f"Task chain created for URL: {url}")
    return task_chain

@celery_app.task
def add(x, y):
    return x + y