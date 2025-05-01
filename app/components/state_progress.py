"""Enhanced state progress visualization component."""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime
import altair as alt
import pandas as pd
from core.state_manager import StateStatus, StateMetadata

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m {seconds % 60:.0f}s"
    hours = minutes / 60
    return f"{hours:.0f}h {minutes % 60:.0f}m"

def display_enhanced_progress(
    state: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None
) -> None:
    """Display enhanced progress visualization.
    
    Args:
        state: Current process state
        history: Optional state history for visualization
    """
    status = state['status']
    metadata = state.get('metadata', {})
    progress = metadata.get('progress', {})
    
    # Create status container
    with st.container():
        # Status header with emoji
        status_emoji = {
            'INIT': '⏳',
            'DOWNLOADING': '⬇️',
            'TRANSCRIBING': '🎯',
            'GENERATING_QUOTES': '✍️',
            'GENERATING_SUMMARY': '📝',
            'COMPLETE': '✅',
            'ERROR': '❌'
        }
        
        st.subheader(f"{status_emoji.get(status, '❓')} {status}")
        
        # Progress visualization
        if isinstance(progress, dict):
            # Overall progress
            total_progress = progress.get('total', 0.0)
            st.progress(total_progress)
            
            # Stage progress
            stage_progress = progress.get('stage_progress', {})
            if stage_progress:
                st.write("Stage Progress:")
                for stage, prog in stage_progress.items():
                    if prog > 0:
                        st.caption(f"{status_emoji.get(stage, '•')} {stage}: {prog*100:.1f}%")
            
            # Estimated time
            est_time = progress.get('estimated_time')
            if est_time:
                st.caption(f"⏱️ Estimated time remaining: {format_duration(est_time)}")
        
        # Error display with retry info
        if status == 'ERROR':
            error = state.get('error', 'Unknown error')
            retries = metadata.get('retries', 0)
            retry_after = metadata.get('retry_after')
            
            with st.error(error):
                if retry_after and retry_after > datetime.now().timestamp():
                    st.caption(f"Retrying... (Attempt {retries})")
                elif retries > 0:
                    st.caption(f"Failed after {retries} retries")
        
        # History visualization if available
        if history:
            st.write("Processing History")
            
            # Create DataFrame for visualization
            df = pd.DataFrame([
                {
                    'timestamp': datetime.fromisoformat(entry['timestamp']),
                    'status': entry['status'],
                    'duration': 0  # Will calculate
                }
                for entry in history
            ])
            
            # Calculate durations
            df['duration'] = (df['timestamp'] - df['timestamp'].shift(1)
                            ).fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
            
            # Create timeline chart
            timeline = alt.Chart(df).mark_bar().encode(
                x='timestamp:T',
                y='duration:Q',
                color='status:N',
                tooltip=['status', 'duration']
            ).properties(
                width=600,
                height=100
            )
            
            st.altair_chart(timeline) 