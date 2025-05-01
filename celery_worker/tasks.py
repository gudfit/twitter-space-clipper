"""Core Celery tasks for processing media files."""
import os
import json
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, cast
from datetime import datetime

from celery import Celery, chain, group  # type: ignore
from pathlib import Path

from celery_worker.celery_app import celery_app
from core.types import ProcessState, create_process_state

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
    
    state = create_process_state(
        stage=stage,
        progress=progress,
        status='processing',
        stage_status=status,
        files=files
    )
    
    save_process_state(storage_dir, space_id, state)

@app.task(bind=True, queue='download')
def download_media(self, url: str, storage_dir: str) -> Dict[str, Any]:
    """Download media from URL."""
    logger.info(f"Starting download task for URL: {url}")
    logger.info(f"Task ID: {self.request.id}")
    logger.info(f"Storage directory: {storage_dir}")
    
    # Import here to avoid circular imports
    from core.processor import get_storage_paths
    from core.download import download_twitter_space
    
    space_id = hashlib.md5(url.encode()).hexdigest()
    storage_paths = get_storage_paths(storage_dir, space_id)
    logger.info(f"Generated space_id: {space_id}")
    logger.info(f"Storage paths: {storage_paths}")
    
    try:
        # Initialize state
        state = create_process_state(
            stage='download',
            stage_status='Starting download...',
            task_id=self.request.id
        )
        
        def progress_callback(d: Dict[str, Any]) -> None:
            if d.get('status') == 'downloading':
                total_bytes = d.get('total_bytes')
                downloaded_bytes = d.get('downloaded_bytes', 0)
                if total_bytes:
                    progress = downloaded_bytes / total_bytes
                    speed = d.get('speed', 0)
                    if speed:
                        speed_mb = speed / (1024 * 1024)
                        status = f"Downloading media: {progress*100:.1f}% ({speed_mb:.1f} MB/s)"
                    else:
                        status = f"Downloading media: {progress*100:.1f}%"
                    
                    # Update state with progress
                    current_state = create_process_state(
                        stage='download',
                        progress=progress * 0.8,  # Leave 20% for post-processing
                        stage_status=status,
                        task_id=self.request.id,
                        files=state.get('files', {})
                    )
                    # Import here to avoid circular imports
                    from core.processor import save_process_state
                    save_process_state(storage_dir, space_id, current_state)
                    state.update(current_state)
            
            elif d.get('status') == 'finished':
                current_console = state.get('console_output', '')
                complete_state = create_process_state(
                    stage='download',
                    progress=1.0,
                    stage_status='Download complete',
                    console_output=f"{current_console}\nDownload complete!" if current_console else "Download complete!",
                    task_id=self.request.id,
                    files=state.get('files', {})
                )
                # Import here to avoid circular imports
                from core.processor import save_process_state
                save_process_state(storage_dir, space_id, complete_state)
                state.update(complete_state)
        
        # Download the media with progress tracking
        download_path = download_twitter_space(url, storage_paths['audio_path'], progress_callback)
        
        # Verify the downloaded file exists and has the correct extension
        if not os.path.exists(download_path):
            # Check for double extension
            potential_path = f"{download_path}.mp3"
            if os.path.exists(potential_path):
                # Rename to correct path
                os.rename(potential_path, download_path)
                logger.info(f"Renamed {potential_path} to {download_path}")
            else:
                raise Exception(f"Downloaded file not found at {download_path} or {potential_path}")
        
        # Update state with completion
        files = {'audio': True, 'transcript': False, 'quotes': False, 'summary': False}
        final_state = create_process_state(
            stage='download',
            progress=1.0,
            stage_status="Download complete",
            console_output=state.get('console_output', ''),
            task_id=self.request.id,
            files=files
        )
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, final_state)
        
        logger.info(f"Download complete for URL: {url}")
        return {
            'space_id': space_id,
            'storage_dir': storage_dir,
            'audio_path': download_path,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        error_state = create_process_state(
            stage='download',
            progress=0.0,
            stage_status="Download failed",
            error=str(e),
            task_id=self.request.id
        )
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, error_state)
        raise

@app.task(bind=True, queue='transcribe')
def transcribe_media(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe downloaded media."""
    logger.info(f"Starting transcription task for space_id: {task_result.get('space_id')}")
    # Import here to avoid circular imports
    from core.processor import get_storage_paths, save_process_state
    from core.transcribe import transcribe_audio
    from core.types import ProcessState
    
    space_id = task_result['space_id']
    storage_dir = task_result['storage_dir']
    audio_path = task_result['audio_path']
    storage_paths = get_storage_paths(storage_dir, space_id)
    
    try:
        # Get current file status
        files = {
            'audio': os.path.exists(storage_paths['audio_path']),
            'transcript': os.path.exists(storage_paths['transcript_path']),
            'quotes': os.path.exists(storage_paths['quotes_path']),
            'summary': os.path.exists(storage_paths['summary_path'])
        }
        
        # Create progress state dict for transcribe_audio
        progress_state: Dict[str, Any] = {'status': '', 'progress': 0.0}
        
        def on_progress_update():
            """Update process state when progress changes."""
            current_state: ProcessState = {
                'status': 'processing',
                'stage': 'transcribe',
                'progress': progress_state['progress'],
                'stage_status': progress_state['status'],
                'last_updated': datetime.now().isoformat(),
                'error': None,
                'files': files,
                'current_chunk': None,
                'total_chunks': None,
                'completed_chunks': [],
                'console_output': None,
                'task_id': self.request.id
            }
            save_process_state(storage_dir, space_id, current_state)
            logger.debug(f"Updated transcribe progress: {progress_state['progress']:.2f} - {progress_state['status']}")
        
        # Set initial progress
        progress_state.update({'status': 'Starting transcription...', 'progress': 0.0})
        on_progress_update()
        
        # Transcribe the audio with progress tracking
        transcript = transcribe_audio(audio_path, progress_callback=progress_state)
        
        # Save transcript
        os.makedirs(os.path.dirname(storage_paths['transcript_path']), exist_ok=True)
        with open(storage_paths['transcript_path'], 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Update files status
        files['transcript'] = True
        
        # Create completion state
        complete_state: ProcessState = {
            'status': 'processing',
            'stage': 'transcribe',
            'progress': 1.0,
            'stage_status': "Transcription complete",
            'last_updated': datetime.now().isoformat(),
            'error': None,
            'files': files,
            'current_chunk': None,
            'total_chunks': None,
            'completed_chunks': [],
            'console_output': None,
            'task_id': self.request.id
        }
        save_process_state(storage_dir, space_id, complete_state)
        
        logger.info(f"Transcription complete for space_id: {space_id}")
        task_result.update({
            'transcript_path': storage_paths['transcript_path'],
            'transcript': transcript
        })
        return task_result
        
    except Exception as e:
        logger.error(f"Transcription failed for space_id {space_id}: {str(e)}")
        # Update state with error but preserve file status
        error_state: ProcessState = {
            'status': 'error',
            'stage': 'transcribe',
            'progress': 0.0,
            'stage_status': str(e),
            'last_updated': datetime.now().isoformat(),
            'error': str(e),
            'files': files,
            'current_chunk': None,
            'total_chunks': None,
            'completed_chunks': [],
            'console_output': None,
            'task_id': self.request.id
        }
        save_process_state(storage_dir, space_id, error_state)
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
        # Create progress callback
        progress_state: Dict[str, Any] = {'status': '', 'progress': 0.0}
        def update_progress():
            current_state: ProcessState = {
                'status': 'processing',
                'stage': 'quotes',
                'progress': progress_state['progress'],
                'stage_status': progress_state['status'],
                'last_updated': datetime.now().isoformat(),
                'error': None,
                'files': {},
                'current_chunk': None,
                'total_chunks': None,
                'completed_chunks': [],
                'console_output': '',
                'task_id': self.request.id
            }
            # Import here to avoid circular imports
            from core.processor import save_process_state
            save_process_state(storage_dir, space_id, current_state)
        
        # Generate quotes with progress tracking
        quotes = create_quote_thread(task_result['transcript'], 
                                   {"url": task_result.get('url')},
                                   progress_callback=progress_state)
        
        # Save quotes
        os.makedirs(os.path.dirname(storage_paths['quotes_path']), exist_ok=True)
        with open(storage_paths['quotes_path'], 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(quotes))
        
        # Update state with completion
        files = {'audio': True, 'transcript': True, 'quotes': True, 'summary': False}
        complete_state: ProcessState = {
            'status': 'processing',
            'stage': 'quotes',
            'progress': 1.0,
            'stage_status': "Quote generation complete",
            'last_updated': datetime.now().isoformat(),
            'error': None,
            'files': files,
            'current_chunk': None,
            'total_chunks': None,
            'completed_chunks': [],
            'console_output': '',
            'task_id': self.request.id
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, complete_state)
        
        logger.info(f"Quote generation complete for space_id: {space_id}")
        task_result.update({
            'quotes_path': storage_paths['quotes_path'],
            'quotes': quotes
        })
        return task_result
    except Exception as e:
        logger.error(f"Quote generation failed for space_id {space_id}: {str(e)}")
        # Update state with error
        state: ProcessState = {
            'status': 'error',
            'stage': 'quotes',
            'progress': 0.0,
            'stage_status': str(e),
            'last_updated': datetime.now().isoformat(),
            'error': str(e),
            'files': {},
            'current_chunk': None,
            'total_chunks': None,
            'completed_chunks': [],
            'console_output': None,
            'task_id': self.request.id
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
        # Create progress callback
        progress_state: Dict[str, Any] = {'status': '', 'progress': 0.0}
        def update_progress():
            current_state: ProcessState = {
                'status': 'processing',
                'stage': 'summary',
                'progress': progress_state['progress'],
                'stage_status': progress_state['status'],
                'last_updated': datetime.now().isoformat(),
                'error': None,
                'files': {},
                'current_chunk': None,
                'total_chunks': None,
                'completed_chunks': [],
                'console_output': '',
                'task_id': self.request.id
            }
            # Import here to avoid circular imports
            from core.processor import save_process_state
            save_process_state(storage_dir, space_id, current_state)
        
        # Generate summary with progress tracking
        summary = generate_summary(task_result['transcript'], 
                                 task_result['quotes'], 
                                 storage_paths['summary_path'],
                                 progress_callback=progress_state)
        
        # Update state with completion
        files = {'audio': True, 'transcript': True, 'quotes': True, 'summary': True}
        final_state: ProcessState = {
            'status': 'complete',
            'stage': 'summary',
            'progress': 1.0,
            'stage_status': "Processing complete",
            'last_updated': datetime.now().isoformat(),
            'error': None,
            'files': files,
            'current_chunk': None,
            'total_chunks': None,
            'completed_chunks': [],
            'console_output': '',
            'task_id': self.request.id
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, final_state)
        
        logger.info(f"Summary generation complete for space_id: {space_id}")
        task_result.update({
            'summary_path': storage_paths['summary_path'],
            'summary': summary
        })
        return task_result
    except Exception as e:
        logger.error(f"Summary generation failed for space_id {space_id}: {str(e)}")
        # Update state with error
        state: ProcessState = {
            'status': 'error',
            'stage': 'summary',
            'progress': 0.0,
            'stage_status': str(e),
            'last_updated': datetime.now().isoformat(),
            'error': str(e),
            'files': {},
            'current_chunk': None,
            'total_chunks': None,
            'completed_chunks': [],
            'console_output': None,
            'task_id': self.request.id
        }
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, state)
        raise

def process_space_chain(url: str, storage_dir: str) -> chain:
    """Create a chain of tasks to process a space."""
    logger.info(f"Creating task chain for URL: {url}")
    
    # Create the chain with immutable signatures to prevent argument chaining issues
    task_chain = chain(
        download_media.si(url, storage_dir),  # Use immutable signature for first task
        transcribe_media.s(),  # Keep mutable for result passing
        generate_quotes_task.s(),  # Keep mutable for result passing
        generate_summary_task.s()  # Keep mutable for result passing
    )
    
    logger.info(f"Task chain created for URL: {url}")
    logger.info(f"Chain tasks: download_media -> transcribe_media -> generate_quotes_task -> generate_summary_task")
    return task_chain

@celery_app.task
def add(x, y):
    return x + y