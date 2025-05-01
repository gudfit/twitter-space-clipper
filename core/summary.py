"""Core functionality for generating summaries from transcripts and quotes."""
import json
import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from utils.api import call_deepseek_api

# Configure module logger
logger = logging.getLogger(__name__)

def extract_from_api_response(response: Optional[Dict[str, Any]], error_msg: str = "Error in API response") -> Tuple[bool, str]:
    """Extract content from API response, handling different response formats.
    
    Args:
        response: The API response dictionary
        error_msg: Custom error message to use if extraction fails
        
    Returns:
        Tuple of (success: bool, content: str)
    """
    try:
        if not response:
            logger.error("No response received from API")
            return False, error_msg
            
        # Handle DeepSeek direct response format
        if 'role' in response and 'content' in response:
            return True, response['content'].strip()
        # Handle format with choices array
        elif 'choices' in response and response['choices']:
            return True, response['choices'][0]['message']['content'].strip()
        else:
            logger.error(f"Unexpected API response format: {response}")
            return False, error_msg
    except Exception as e:
        logger.error(f"Error extracting from API response: {str(e)}")
        return False, error_msg

def split_into_sections(transcript: str, max_length: int = 2000) -> List[str]:
    """Split transcript into manageable sections."""
    # Split by paragraphs first
    paragraphs = transcript.split('\n\n')
    sections: List[str] = []
    current_section: List[str] = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        if current_length + para_length > max_length and current_section:
            sections.append('\n\n'.join(current_section))
            current_section = [para]
            current_length = para_length
        else:
            current_section.append(para)
            current_length += para_length
    
    if current_section:
        sections.append('\n\n'.join(current_section))
    
    return sections

def summarize_section(section: str) -> str:
    """Generate a summary for a section of text."""
    response = call_deepseek_api([
        {"role": "system", "content": """You are an expert at summarizing content. Create clear, concise summaries that capture the key points while maintaining readability."""},
        {"role": "user", "content": f"Summarize this section of text in a few sentences:\n\n{section}"}
    ])
    
    success, content = extract_from_api_response(response, "Error generating section summary")
    return content

def analyze_quotes(quotes: List[str]) -> str:
    """Analyze quotes to extract key insights."""
    if not quotes:
        return "No quotes available for analysis"
    
    quotes_text = '\n'.join(quotes)
    response = call_deepseek_api([
        {"role": "system", "content": """You are an expert at analyzing quotes and extracting key themes and insights. Focus on identifying the main points and recurring themes."""},
        {"role": "user", "content": f"Analyze these quotes and identify the key themes and insights:\n\n{quotes_text}"}
    ])
    
    success, content = extract_from_api_response(response, "Error analyzing quotes")
    return content

def combine_summaries(section_summaries: List[str], quote_insights: str) -> str:
    """Combine section summaries and quote insights into a final summary."""
    combined_text = '\n\n'.join(section_summaries) + '\n\nQuote Analysis:\n' + quote_insights
    
    response = call_deepseek_api([
        {"role": "system", "content": """You are an expert at synthesizing information into clear, comprehensive summaries. Create a cohesive overview that combines multiple sources of information."""},
        {"role": "user", "content": f"Create a comprehensive summary combining these section summaries and quote analysis:\n\n{combined_text}"}
    ])
    
    success, content = extract_from_api_response(response, "Error generating final summary")
    return content

def extract_key_points(summary: str) -> List[str]:
    """Extract key points from the summary."""
    response = call_deepseek_api([
        {"role": "system", "content": """You are an expert at identifying and extracting key points from text. Focus on the most important takeaways."""},
        {"role": "user", "content": f"Extract 3-5 key points from this summary. Return only the points, one per line:\n\n{summary}"}
    ])
    
    success, content = extract_from_api_response(response, "Error extracting key points")
    if not success:
        return ["Error extracting key points"]
        
    # Split into points and clean up
    points = content.split('\n')
    return [p.strip('- ') for p in points if p.strip()]

def generate_summary(transcript: str, quotes: List[str], output_path: str, progress_callback: Optional[Dict[str, Any]] = None) -> Dict[str, Union[str, List[str]]]:
    """Generate a summary of the transcript and quotes."""
    logger.info("Starting summary generation")
    
    try:
        if not transcript or not transcript.strip():
            error_msg = "Empty transcript provided"
            logger.error(error_msg)
            if progress_callback:
                progress_callback.update({
                    'status': error_msg,
                    'progress': 0.0
                })
            initial_error_summary: Dict[str, Union[str, List[str]]] = {
                'overview': error_msg,
                'key_points': ["No content to summarize"]
            }
            save_summary(initial_error_summary, output_path)
            return initial_error_summary
        
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
            
            summary_text = summarize_section(section)
            if summary_text.startswith("Error"):
                logger.error(f"Error summarizing section {i+1}: {summary_text}")
                continue
                
            section_summaries.append(summary_text)
            logger.info(f"Summarized section {i+1}/{total_sections}")
        
        if not section_summaries:
            error_msg = "Failed to generate any section summaries"
            logger.error(error_msg)
            if progress_callback:
                progress_callback.update({
                    'status': error_msg,
                    'progress': 0.0
                })
            section_error_summary: Dict[str, Union[str, List[str]]] = {
                'overview': error_msg,
                'key_points': ["Error generating summary"]
            }
            save_summary(section_error_summary, output_path)
            return section_error_summary
        
        # Analyze quotes if available
        if progress_callback:
            progress_callback.update({
                'status': 'Analyzing content...',
                'progress': 0.7
            })
        
        # If no quotes available, just use the section summaries
        if not quotes:
            logger.warning("No quotes available for analysis, generating summary from transcript only")
            quote_insights = "No notable quotes were extracted from the content."
        else:
            quote_insights = analyze_quotes(quotes)
        
        # Generate final summary
        if progress_callback:
            progress_callback.update({
                'status': 'Generating final summary...',
                'progress': 0.8
            })
        
        final_summary = combine_summaries(section_summaries, quote_insights)
        
        # Create summary dict
        result_summary: Dict[str, Union[str, List[str]]] = {
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
        save_summary(result_summary, output_path)
        
        if progress_callback:
            progress_callback.update({
                'status': 'Summary generation complete',
                'progress': 1.0
            })
        
        return result_summary
        
    except Exception as e:
        error_msg = f"Error generating summary: {str(e)}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback.update({
                'status': error_msg,
                'progress': 0.0
            })
        final_error_summary: Dict[str, Union[str, List[str]]] = {
            'overview': error_msg,
            'key_points': ["Error extracting key points"]
        }
        save_summary(final_error_summary, output_path)
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