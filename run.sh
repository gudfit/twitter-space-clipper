#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add the project root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Change to the project root directory
cd "${SCRIPT_DIR}"

# Run Streamlit with no-watchdog flag
streamlit run app/main.py --server.fileWatcherType none
