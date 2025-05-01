"""Core functionality for generating summaries from transcripts and quotes."""
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from utils.api import call_deepseek_api

# Configure module logger
logger = logging.getLogger(__name__)

def generate_summary(transcript: str, quotes: List[str], output_path: str, progress_callback: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, List[str]]]:
    """Generate a summary of the transcript and quotes."""
    logger.info("Starting summary generation")
    
    try:
        if progress_callback:
            progress_callback.update({
                'status': 'Analyzing transcript...',
                'progress': 0.1
            })
        
        # Split transcript into sections
        sections = split_into_sections(transcript)
        total_sections = len(sections)
        
        if progress_callback:
            progress_callback.update({
                'status': f'Processing {total_sections} sections...',
                'progress': 0.2
            })
        
        # Generate section summaries
        section_summaries = []
        for i, section in enumerate(sections):
            if progress_callback:
                progress = 0.2 + (0.4 * (i / total_sections))  # Scale from 20% to 60%
                progress_callback.update({
                    'status': f'Summarizing section {i+1}/{total_sections}',
                    'progress': progress
                })
            
            summary = summarize_section(section)
            section_summaries.append(summary)
            
            # Log progress
            logger.info(f"Summarized section {i+1}/{total_sections}")
        
        # Analyze quotes
        if progress_callback:
            progress_callback.update({
                'status': 'Analyzing quotes...',
                'progress': 0.7
            })
        
        quote_insights = analyze_quotes(quotes)
        
        # Generate final summary
        if progress_callback:
            progress_callback.update({
                'status': 'Generating final summary...',
                'progress': 0.8
            })
        
        final_summary = combine_summaries(section_summaries, quote_insights)
        
        # Create summary dict
        summary = {
            'overview': final_summary,
            'key_points': extract_key_points(final_summary)
        }
        
        # Save summary
        if progress_callback:
            progress_callback.update({
                'status': 'Saving summary...',
                'progress': 0.9
            })
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_summary(summary, output_path)
        
        if progress_callback:
            progress_callback.update({
                'status': 'Summary generation complete',
                'progress': 1.0
            })
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        if progress_callback:
            progress_callback.update({
                'status': f'Error: {str(e)}',
                'progress': 0.0
            })
        raise

def save_summary(summary: Dict[str, Union[str, List[str]]], output_path: str) -> None:
    """Save summary to a JSON file.
    
    Args:
        summary: Dictionary containing summary data
        output_path: Path to save the JSON file
        
    Raises:
        IOError: If there's an error writing to the file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {output_path}")
    except IOError as e:
        logger.error(f"Error saving summary to {output_path}: {e}")
        raise

def load_summary(summary_path: str) -> Dict[str, Union[str, List[str]]]:
    """Load summary from a JSON file.
    
    Args:
        summary_path: Path to the JSON file to load
        
    Returns:
        Dictionary containing summary data or None if file doesn't exist
        
    Raises:
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        logger.info(f"Summary loaded from {summary_path}")
        return summary
    except FileNotFoundError:
        logger.warning(f"Summary file not found: {summary_path}")
        return {
            "overview": "Error loading summary: File not found",
            "key_points": []
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding summary JSON from {summary_path}: {e}")
        raise 