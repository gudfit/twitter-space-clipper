"""Core Celery tasks for processing media files."""
import os
import json
import logging
import time
import hashlib
import socket
from typing import Dict, List, Optional, Any, cast
from datetime import datetime

from celery import Celery, chain, group  # type: ignore
from pathlib import Path

from celery_worker.celery_app import celery_app
from core.types import ProcessState, create_process_state
from core.hostname import HOSTNAME, get_namespaced_key
from core.processor import save_process_state, get_process_state
from core.download import download_twitter_space
from core.transcribe import transcribe_audio
from core.quotes import create_quote_thread
from core.summary import generate_summary
from core.redis_manager import RedisManager
from core.state_manager import StateManager

# Configure module logger
logger = logging.getLogger(__name__)

# Initialize Redis and state manager
redis_client = RedisManager.get_client()
state_manager = StateManager(redis_client)

# Initialize Celery
app = celery_app

def get_namespaced_task_id(task_id: str) -> str:
    """Get a hostname-namespaced task ID."""
    return get_namespaced_key('task', task_id)

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
        files=files,
        hostname=HOSTNAME
    )
    
    save_process_state(storage_dir, space_id, state)

@app.task(bind=True, queue='download')
def download_media(self, url: str, storage_dir: str) -> Dict[str, Any]:
    """Download media from URL."""
    task_id = get_namespaced_task_id(self.request.id)
    logger.info(f"Starting download task for URL: {url}")
    logger.info(f"Task ID: {task_id}")
    logger.info(f"Storage directory: {storage_dir}")
    
    # Import here to avoid circular imports
    from core.processor import get_storage_paths
    from core.download import download_twitter_space
    
    space_id = get_namespaced_key('space', hashlib.md5(url.encode()).hexdigest())
    storage_paths = get_storage_paths(storage_dir, space_id)
    logger.info(f"Generated space_id: {space_id}")
    logger.info(f"Storage paths: {storage_paths}")
    
    try:
        # First check if MP3 already exists and is valid
        if os.path.exists(storage_paths['audio_path']) and os.path.getsize(storage_paths['audio_path']) > 0:
            logger.info(f"✓ MP3 file already exists at {storage_paths['audio_path']}")
            # Return success with existing file
            return {
                'space_id': space_id,
                'storage_dir': storage_dir,
                'audio_path': storage_paths['audio_path'],
                'success': True
            }

        # Initialize state
        state = create_process_state(
            stage='download',
            stage_status='Starting download...',
            task_id=task_id,
            hostname=HOSTNAME
        )
        
        # Import here to avoid circular imports
        from core.processor import save_process_state
        save_process_state(storage_dir, space_id, state)
        
        # Download the file
        logger.info("⏳ Starting media download...")
        download_path = download_twitter_space(url, storage_paths['audio_path'])
        
        if not download_path or not os.path.exists(download_path):
            raise Exception("Download failed - no output file produced")
            
        logger.info(f"✓ Download complete: {download_path}")
        
        # Get current file status
        files = {
            'audio': os.path.exists(storage_paths['audio_path']),
            'transcript': os.path.exists(storage_paths['transcript_path']),
            'quotes': os.path.exists(storage_paths['quotes_path']),
            'summary': os.path.exists(storage_paths['summary_path'])
        }
        
        final_state = create_process_state(
            stage='download',
            progress=1.0,
            stage_status="Download complete",
            console_output=state.get('console_output', ''),
            task_id=task_id,
            files=files,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, final_state)
        
        logger.info("✓ Download task complete")
        return {
            'space_id': space_id,
            'storage_dir': storage_dir,
            'audio_path': download_path,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Download failed: {str(e)}")
        error_state = create_process_state(
            stage='download',
            progress=0.0,
            stage_status="Download failed",
            error=str(e),
            task_id=task_id,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, error_state)
        raise

@app.task(bind=True, queue='transcribe')
def transcribe_media(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe downloaded media."""
    task_id = get_namespaced_task_id(self.request.id)
    space_id = task_result['space_id']
    logger.info(f"Starting transcription task for space_id: {space_id}")
    logger.info(f"Task ID: {task_id}")
    
    # Import here to avoid circular imports
    from core.processor import get_storage_paths, save_process_state
    from core.transcribe import transcribe_audio
    from core.types import ProcessState
    
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
        
        def update_progress():
            current_state = create_process_state(
                stage='transcribe',
                progress=progress_state['progress'],
                stage_status=progress_state['status'],
                task_id=task_id,
                hostname=HOSTNAME,
                files=files
            )
            save_process_state(storage_dir, space_id, current_state)
            logger.info(f"⏳ Transcription progress: {progress_state['status']} ({progress_state['progress']*100:.1f}%)")
        
        # Bind progress callback
        progress_state['callback'] = update_progress
        
        # Start transcription
        logger.info("⏳ Starting audio transcription...")
        transcript = transcribe_audio(audio_path, progress_state)
        
        if not transcript:
            raise Exception("Transcription failed - no output produced")
            
        # Save transcript
        os.makedirs(os.path.dirname(storage_paths['transcript_path']), exist_ok=True)
        with open(storage_paths['transcript_path'], 'w', encoding='utf-8') as f:
            f.write(transcript)
            
        logger.info(f"✓ Transcription saved to {storage_paths['transcript_path']}")
        
        # Update state with completion
        files['transcript'] = True
        complete_state = create_process_state(
            stage='transcribe',
            progress=1.0,
            stage_status="Transcription complete",
            files=files,
            task_id=task_id,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, complete_state)
        
        logger.info("✓ Transcription task complete")
        task_result.update({
            'transcript_path': storage_paths['transcript_path'],
            'transcript': transcript
        })
        return task_result
        
    except Exception as e:
        logger.error(f"❌ Transcription failed: {str(e)}")
        error_state = create_process_state(
            stage='transcribe',
            progress=0.0,
            stage_status=str(e),
            error=str(e),
            files=files,
            task_id=task_id,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, error_state)
        raise

@app.task(bind=True, queue='generate')
def generate_quotes_task(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate quotes from transcript."""
    task_id = get_namespaced_task_id(self.request.id)
    space_id = task_result['space_id']
    logger.info(f"Starting quote generation for space_id: {space_id}")
    logger.info(f"Task ID: {task_id}")
    
    # Import here to avoid circular imports
    from core.processor import get_storage_paths, save_process_state
    from core.quotes import create_quote_thread
    
    storage_dir = task_result['storage_dir']
    storage_paths = get_storage_paths(storage_dir, space_id)
    
    try:
        # Ensure quotes directory exists
        quotes_dir = os.path.dirname(storage_paths['quotes_path'])
        if not os.path.exists(quotes_dir):
            logger.info(f"Creating quotes directory: {quotes_dir}")
            os.makedirs(quotes_dir, exist_ok=True)
        
        # Create progress callback
        progress_state: Dict[str, Any] = {'status': '', 'progress': 0.0}
        def update_progress():
            current_state = create_process_state(
                stage='quotes',
                progress=progress_state['progress'],
                stage_status=progress_state['status'],
                files={
                    'audio': os.path.exists(storage_paths['audio_path']),
                    'transcript': os.path.exists(storage_paths['transcript_path']),
                    'quotes': os.path.exists(storage_paths['quotes_path']),
                    'summary': os.path.exists(storage_paths['summary_path'])
                },
                task_id=task_id,
                hostname=HOSTNAME
            )
            save_process_state(storage_dir, space_id, current_state)
            logger.info(f"Quote generation progress: {progress_state['status']} ({progress_state['progress']*100:.1f}%)")
        
        # Validate transcript
        if not task_result.get('transcript'):
            error_msg = "No transcript provided in task result"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Check transcript length
        transcript = task_result['transcript']
        logger.info(f"Transcript length: {len(transcript)} chars")
        if len(transcript.strip()) == 0:
            error_msg = "Empty transcript provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Generate quotes with progress tracking
        logger.info("Starting quote generation...")
        progress_state['status'] = 'Generating quotes...'
        progress_state['progress'] = 0.1
        update_progress()
        
        # Get URL from task result or use a default
        url = task_result.get('url', 'Unknown URL')
        logger.info(f"Processing URL: {url}")
        
        # Generate quotes
        quotes = create_quote_thread(
            transcript, 
            {"url": url},
            progress_callback=progress_state
        )
        
        if not quotes:
            error_msg = "Quote generation failed - no quotes produced"
            logger.error(error_msg)
            
            # Save empty quotes file to indicate completion
            try:
                with open(storage_paths['quotes_path'], 'w', encoding='utf-8') as f:
                    f.write('')
                logger.info("Created empty quotes file to indicate completion")
            except IOError as e:
                logger.error(f"Failed to create empty quotes file: {e}")
            
            # Update state with error
            error_state = create_process_state(
                stage='quotes',
                progress=0.0,
                stage_status=error_msg,
                error=error_msg,
                files={
                    'audio': os.path.exists(storage_paths['audio_path']),
                    'transcript': os.path.exists(storage_paths['transcript_path']),
                    'quotes': True,  # File exists but empty
                    'summary': False
                },
                task_id=task_id,
                hostname=HOSTNAME
            )
            save_process_state(storage_dir, space_id, error_state)
            raise Exception(error_msg)
            
        logger.info(f"Generated {len(quotes)} quotes")
        
        # Update progress before saving
        progress_state['status'] = 'Saving quotes...'
        progress_state['progress'] = 0.9
        update_progress()
        
        # Save quotes with error handling
        try:
            logger.info(f"Saving quotes to {storage_paths['quotes_path']}")
            with open(storage_paths['quotes_path'], 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(quotes))
        except IOError as e:
            error_msg = f"Failed to save quotes file: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        logger.info(f"Quotes saved successfully")
        
        # Update state with completion
        complete_state = create_process_state(
            stage='quotes',
            progress=1.0,
            stage_status="Quote generation complete",
            files={
                'audio': True, 
                'transcript': True, 
                'quotes': True, 
                'summary': False
            },
            task_id=task_id,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, complete_state)
        
        logger.info("Quote generation task complete")
        task_result.update({
            'quotes_path': storage_paths['quotes_path'],
            'quotes': quotes
        })
        return task_result
        
    except Exception as e:
        error_msg = f"Quote generation failed: {str(e)}"
        logger.exception(error_msg)
        error_state = create_process_state(
            stage='quotes',
            progress=0.0,
            stage_status=error_msg,
            error=error_msg,
            files={
                'audio': os.path.exists(storage_paths['audio_path']),
                'transcript': os.path.exists(storage_paths['transcript_path']),
                'quotes': False,
                'summary': False
            },
            task_id=task_id,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, error_state)
        raise

@app.task(bind=True, queue='generate')
def generate_summary_task(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary from transcript and quotes."""
    task_id = get_namespaced_task_id(self.request.id)
    space_id = task_result['space_id']
    logger.info(f"Starting summary generation for space_id: {space_id}")
    logger.info(f"Task ID: {task_id}")
    
    # Import here to avoid circular imports
    from core.processor import get_storage_paths
    from core.summary import generate_summary
    
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
                'task_id': task_id,
                'hostname': HOSTNAME
            }
            save_process_state(storage_dir, space_id, current_state)
            logger.info(f"⏳ Summary generation progress: {progress_state['status']} ({progress_state['progress']*100:.1f}%)")
        
        # Generate summary with progress tracking
        logger.info("⏳ Starting summary generation...")
        summary = generate_summary(task_result['transcript'], 
                                 task_result['quotes'], 
                                 storage_paths['summary_path'],
                                 progress_callback=progress_state)
        
        if not summary:
            raise Exception("Summary generation failed - no summary produced")
            
        logger.info(f"✓ Summary saved to {storage_paths['summary_path']}")
        
        # Update state with completion
        files = {'audio': True, 'transcript': True, 'quotes': True, 'summary': True}
        final_state = create_process_state(
            stage='summary',
            progress=1.0,
            stage_status="Processing complete",
            status='complete',
            files=files,
            task_id=task_id,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, final_state)
        
        logger.info("✓ Summary generation task complete")
        task_result.update({
            'summary_path': storage_paths['summary_path'],
            'summary': summary
        })
        return task_result
        
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {str(e)}")
        error_state = create_process_state(
            stage='summary',
            progress=0.0,
            stage_status=str(e),
            error=str(e),
            files={},
            task_id=task_id,
            hostname=HOSTNAME
        )
        save_process_state(storage_dir, space_id, error_state)
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