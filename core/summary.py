"""Core functionality for generating summaries from transcripts and quotes."""
import json
import logging
from typing import Dict, List, Optional, Any
from utils.api import call_deepseek_api

# Configure module logger
logger = logging.getLogger(__name__)

def generate_summary(transcript: str, quotes: List[str]) -> Dict[str, str]:
    """Generate a comprehensive summary using both transcript and quotes.
    
    Args:
        transcript: The full transcript text to summarize
        quotes: List of key quotes extracted from the transcript
        
    Returns:
        Dict containing 'overview' (str) and 'key_points' (List[str])
        
    Example:
        >>> summary = generate_summary(transcript, quotes)
        >>> print(summary['overview'])
        >>> for point in summary['key_points']:
        >>>     print(f"- {point}")
    """
    logger.info("Starting summary generation...")
    
    try:
        # First, get a high-level summary from DeepSeek
        response = call_deepseek_api([
            {"role": "system", "content": """You are an expert at summarizing content. Create a clear, engaging summary that captures the main points and key insights. Focus on providing value to someone who hasn't heard the original content. Keep formatting minimal and clean."""},
            {"role": "user", "content": f"""Please summarize this content with:
1. A brief overview paragraph (2-3 sentences)
2. 4-6 key points as simple bullet points

Here's the transcript:
{transcript}

And here are some key quotes that were identified:
{chr(10).join(quotes)}"""}
        ])
        
        if not response:
            logger.error("No response received from DeepSeek API")
            return {
                "overview": "Error generating summary: No response from API",
                "key_points": []
            }

        # Parse the response into overview and key points
        content = response['content']
        logger.debug(f"Received content of length: {len(content)}")
        
        # Split into sections and clean up
        sections = content.split('\n\n')
        overview = sections[0].strip()
        
        # Extract bullet points (looking for lines starting with • or -)
        key_points = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('•') or line.startswith('-'):
                point = line.lstrip('•- ').strip()
                if point:
                    key_points.append(point)

        if not key_points:
            logger.warning("No key points extracted from summary")
            
        logger.info(f"Successfully generated summary with {len(key_points)} key points")
        return {
            "overview": overview,
            "key_points": key_points
        }
        
    except Exception as e:
        logger.exception("Error generating summary")
        return {
            "overview": f"Error generating summary: {str(e)}",
            "key_points": []
        }

def save_summary(summary: Dict[str, Any], path: str) -> None:
    """Save summary to a JSON file.
    
    Args:
        summary: Dictionary containing summary data
        path: Path to save the JSON file
        
    Raises:
        IOError: If there's an error writing to the file
    """
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to {path}")
    except IOError as e:
        logger.error(f"Error saving summary to {path}: {e}")
        raise

def load_summary(path: str) -> Optional[Dict[str, Any]]:
    """Load summary from a JSON file.
    
    Args:
        path: Path to the JSON file to load
        
    Returns:
        Dictionary containing summary data or None if file doesn't exist
        
    Raises:
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        logger.info(f"Summary loaded from {path}")
        return summary
    except FileNotFoundError:
        logger.warning(f"Summary file not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding summary JSON from {path}: {e}")
        raise 