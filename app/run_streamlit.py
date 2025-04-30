"""Wrapper script to start Streamlit."""
import os
import sys
from streamlit.web.cli import main
import streamlit.watcher.path_watcher

# Patch the file watcher to ignore PyTorch files
original_watch_file = streamlit.watcher.path_watcher.watch_file

def patched_watch_file(filepath: str, *args, **kwargs):
    """Patched watch_file function that ignores PyTorch files."""
    if 'torch' in filepath or '_C' in filepath:
        return None
    return original_watch_file(filepath, *args, **kwargs)

streamlit.watcher.path_watcher.watch_file = patched_watch_file

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_app = os.path.join(script_dir, "main.py")
    
    # Set up Streamlit command line args
    # --server.fileWatcherType none disables the file watcher
    sys.argv = ["streamlit", "run", 
                "--server.fileWatcherType", "none",
                main_app]
    
    # Start Streamlit using CLI
    main() 