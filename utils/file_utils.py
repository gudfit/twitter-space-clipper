"""Utilities for file handling and naming."""
import re
from datetime import datetime
import os
from typing import Dict, Optional

def clean_filename(title: str) -> str:
    """Clean filename by removing special characters and limiting length.
    
    Args:
        title: The title to clean
        
    Returns:
        A cleaned filename safe for filesystem use
    """
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'[-\s]+', '_', clean)
    return clean[:10]

def format_output_filename(title: str, output_type: str) -> str:
    """Format output filename according to requirements.
    
    Args:
        title: The base title for the file
        output_type: The type of output (e.g., 'transcript', 'summary')
        
    Returns:
        A formatted filename with date stamp
    """
    clean_title = clean_filename(title)[:10].strip()
    date_str = datetime.now().strftime("%m%d%H%M")
    return f"{clean_title}_{date_str}_{output_type}.txt"

def get_project_dir(base_dir: str, title: str) -> str:
    """Create and return project-specific directory based on title.
    
    Args:
        base_dir: The base directory to create the project dir in
        title: The title to use for the project directory
        
    Returns:
        Path to the project directory
    """
    project_name = clean_filename(title)[:10].strip()
    project_dir = os.path.join(base_dir, project_name)
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

def normalize_space_info(space_info: Optional[Dict]) -> Dict:
    """Normalize space info to a flat structure for compatibility.
    
    Args:
        space_info: Raw space info dictionary
        
    Returns:
        Normalized space info with consistent structure
    """
    if not space_info:
        return {}
        
    normalized = {
        'title': space_info.get('title'),
        'url': space_info.get('url')
    }
    
    # Handle host info
    if host := space_info.get('host'):
        if isinstance(host, dict):
            normalized['host'] = host.get('handle')
            normalized['host_display_name'] = host.get('name')
        else:
            normalized['host'] = str(host)
    
    # Handle speakers info
    if speakers := space_info.get('speakers'):
        if speakers and isinstance(speakers[0], dict):
            normalized['speakers'] = [s.get('handle') for s in speakers]
            normalized['speaker_display_names'] = {
                s.get('handle'): s.get('name') 
                for s in speakers 
                if s.get('handle')
            }
        else:
            normalized['speakers'] = [str(s) for s in speakers]
    
    return normalized 