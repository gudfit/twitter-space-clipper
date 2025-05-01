"""Component for displaying process state."""

import streamlit as st
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path
from core.state_manager import StateStatus, StateMetadata
from core.types import StoragePaths

def display_state(state: Dict[str, Any]) -> None:
    """Display current process state.
    
    Args:
        state: Current process state
    """
    status = state['status']
    message = state['message']
    error = state['error']
    
    if status == 'processing':
        with st.status(message, expanded=True) as status:
            # Show current stage
            st.write(message)
            
            # Show task ID if available
            task_id = state.get('metadata', {}).get('task_id')
            if task_id:
                st.caption(f"Task ID: {task_id}")
                
    elif status == 'complete':
        st.success(message)
        
    elif status == 'error':
        st.error(f"{message}: {error}")
        
    elif status == 'unknown':
        st.warning(message)
        
def display_file_status(
    space_id: str,
    storage_paths: StoragePaths,
    show_buttons: bool = True,
    on_action: Optional[Callable[[str, str, str], None]] = None
) -> None:
    """Display status of output files.
    
    Args:
        space_id: Space identifier
        storage_paths: Paths to output files
        show_buttons: Whether to show action buttons
        on_action: Callback for file actions (space_id, file_type, action)
    """
    # Check file existence
    files_exist = {
        'Audio': storage_paths['audio_path'],
        'Transcript': storage_paths['transcript_path'],
        'Quotes': storage_paths['quotes_path'],
        'Summary': storage_paths['summary_path']
    }
    
    # Create columns for each file
    cols = st.columns(len(files_exist))
    
    for col, (file_type, path) in zip(cols, files_exist.items()):
        with col:
            exists = path and Path(str(path)).exists() and Path(str(path)).stat().st_size > 0
            
            if exists:
                st.success(f"{file_type} ✓")
                if show_buttons:
                    if st.button(f"View {file_type}", key=f"view_{file_type}_{space_id}"):
                        if on_action:
                            on_action(space_id, file_type, 'view')
            else:
                st.error(f"{file_type} ✗")
                if show_buttons and st.button(f"Generate {file_type}", key=f"generate_{file_type}_{space_id}"):
                    if on_action:
                        on_action(space_id, file_type, 'generate')

def display_metadata(metadata: Union[StateMetadata, Dict[str, Any]]) -> None:
    """Display process metadata.
    
    Args:
        metadata: Process metadata
    """
    with st.expander("Process Details"):
        # Show URL
        url = metadata.get('url')
        if url:
            st.write("**URL:**", url)
        
        # Show space ID
        space_id = metadata.get('space_id')
        if space_id:
            st.write("**Space ID:**", space_id)
        
        # Show task ID
        task_id = metadata.get('task_id')
        if task_id:
            st.write("**Task ID:**", task_id)
        
        # Show timestamps
        updated_at = metadata.get('updated_at')
        if updated_at:
            st.write("**Last Updated:**", updated_at) 