"""Wrapper script to start Streamlit."""
import os
import sys
from streamlit.web.cli import main

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