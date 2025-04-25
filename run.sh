#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Streamlit with no-watchdog flag
streamlit run app.py --server.fileWatcherType none
