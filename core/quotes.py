import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from dotenv import load_dotenv
import time
from utils.api import call_deepseek_api
import logging

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

def clean_quote(quote: str) -> str:
    """Clean and format a quote for Twitter"""
    # Remove extra whitespace and newlines
    quote = ' '.join(quote.split())
    # Ensure proper spacing after punctuation
    quote = re.sub(r'([.,!?])(\w)', r'\1 \2', quote)
    # Remove duplicate spaces
    quote = re.sub(r'\s+', ' ', quote)
    return quote.strip()

def find_speaker_in_text(text: str, speaker: str) -> Optional[str]:
    """Find speaker mentions using fuzzy matching"""
    # Look for exact handle match first
    handle_pattern = f"@?{re.escape(speaker)}\\b"
    if re.search(handle_pattern, text, re.IGNORECASE):
        return speaker
    
    return None

def parse_transcript_segments(transcript: str, speakers: List[str]) -> List[Dict]:
    """Parse transcript into speaker segments with improved detection"""
    segments = []
    current_speaker = None
    current_text = []
    
    # Common patterns in Whisper transcripts
    speaker_patterns = [
        r'^([^:]+):\s*(.+)',  # Basic pattern: "Speaker: text"
        r'^(.*?(?:says|said|asks|asked|continues|continued|explains|explained))\s*[,:]?\s*(.+)',  # Speaking verbs
        r'^\[([^\]]+)\]\s*(.+)'  # [Speaker] text
    ]
    
    # Split transcript into paragraphs
    paragraphs = [p.strip() for p in transcript.split('\n\n') if p.strip()]
    
    for paragraph in paragraphs:
        # Try to identify speaker at start of paragraph
        speaker_found = False
        paragraph_text = paragraph.strip()
        
        # Try each pattern to find speaker
        for pattern in speaker_patterns:
            match = re.match(pattern, paragraph_text, re.IGNORECASE)
            if match:
                potential_speaker = match.group(1).strip()
                text = match.group(2).strip()
                
                # Check if potential speaker matches any known speaker
                for speaker in speakers:
                    if potential_speaker.lower() == speaker.lower():
                        
                        # Save previous segment if exists
                        if current_speaker and current_text:
                            segments.append({
                                'speaker': current_speaker,
                                'text': ' '.join(current_text),
                                'context': paragraph
                            })
                        
                        # Start new segment
                        current_speaker = speaker
                        current_text = [text]
                        speaker_found = True
                        break
                
                if speaker_found:
                    break
        
        # If no speaker found, append to current segment if exists
        if not speaker_found and current_speaker:
            current_text.append(paragraph_text)
    
    # Add final segment
    if current_speaker and current_text:
        segments.append({
            'speaker': current_speaker,
            'text': ' '.join(current_text),
            'context': ' '.join(current_text)
        })
    
    return segments

def chunk_transcript(transcript: str, chunk_size: int = 50000) -> List[str]:
    """Split transcript into manageable chunks while preserving sentence boundaries."""
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0
    
    # Split by sentences (roughly)
    sentences = re.split(r'(?<=[.!?])\s+', transcript)
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > chunk_size and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def extract_quotes_from_response(response: Dict[str, Any], chunk_num: Optional[int] = None) -> List[str]:
    """Extract quotes from API response, handling different response formats."""
    try:
        # Handle DeepSeek direct response format
        if 'role' in response and 'content' in response:
            quotes_text = response['content'].strip()
        # Handle format with choices array
        elif 'choices' in response and response['choices']:
            quotes_text = response['choices'][0]['message']['content'].strip()
        else:
            chunk_info = f" for chunk {chunk_num}" if chunk_num is not None else ""
            logger.error(f"Unexpected API response format{chunk_info}: {response}")
            return []
            
        if not quotes_text:
            chunk_info = f" for chunk {chunk_num}" if chunk_num is not None else ""
            logger.warning(f"Empty quotes response from API{chunk_info}")
            return []
            
        # Split into individual quotes and clean up
        quotes = []
        raw_quotes = quotes_text.split('\n')
        logger.info(f"Processing {len(raw_quotes)} raw quotes")
        
        for quote in raw_quotes:
            quote = quote.strip()
            if not quote:
                continue
                
            # Remove surrounding quotes if present
            quote = quote.strip('"')
            
            # Clean and validate quote
            quote = clean_quote(quote)
            if len(quote) <= 240:  # Twitter limit
                quotes.append(quote)
                logger.info(f"Added quote ({len(quote)} chars): {quote[:50]}...")
            else:
                logger.warning(f"Quote too long ({len(quote)} chars), skipping: {quote[:50]}...")
        
        return quotes
    except Exception as e:
        chunk_info = f" for chunk {chunk_num}" if chunk_num is not None else ""
        logger.error(f"Error processing quotes{chunk_info}: {str(e)}")
        return []

def extract_speaker_quotes(transcript: str, speaker_info: Dict) -> List[str]:
    """Extract the best quotes for each speaker using DeepSeek"""
    try:
        # If no speaker info, just extract general quotes
        if not speaker_info or not speaker_info.get('speaker'):
            print(f"ℹ️ No speaker information provided, extracting general quotes...")
            return extract_general_quotes(transcript)
            
        speaker = speaker_info.get('speaker')
        print(f"\n🔍 Analyzing transcript for quotes from {speaker}...")
        
        # Parse transcript into segments
        all_speakers = []
        if speaker_info.get('space_info'):
            if speaker_info['space_info'].get('host'):
                all_speakers.append(speaker_info['space_info']['host'])
            if speaker_info['space_info'].get('speakers'):
                all_speakers.extend(speaker_info['space_info']['speakers'])
        
        if not all_speakers:
            print("ℹ️ No speaker information available, extracting general quotes")
            return extract_general_quotes(transcript)
            
        segments = parse_transcript_segments(transcript, all_speakers)
        if not segments:
            print(f"ℹ️ No segments found for {speaker}, extracting general quotes")
            return extract_general_quotes(transcript)
            
        # Filter segments for this speaker
        speaker_segments = [s for s in segments if s['speaker'].lower() == speaker.lower()]
        if not speaker_segments:
            print(f"ℹ️ No segments found for {speaker}, extracting general quotes")
            return extract_general_quotes(transcript)
            
        # Process segments in chunks to avoid API limits
        all_quotes = []
        chunks = chunk_transcript('\n\n'.join([s['text'] for s in speaker_segments]))
        
        print(f"\n📊 Processing {len(chunks)} chunks of transcript...")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n🔄 Processing chunk {i}/{len(chunks)}...")
            
            # Ask DeepSeek to extract quotes from this chunk
            response = call_deepseek_api([
                {"role": "system", "content": """You are an expert at identifying impactful and quotable moments from conversations. Your task is to extract the most interesting, insightful, or memorable quotes that would work well on Twitter. Each quote should:
1. Be self-contained and meaningful on its own
2. Be around 240 characters
3. Capture a key insight, opinion, or memorable statement
4. Be something worth quoting

Format each quote on a new line.
Do not include any summary, heading, or commentary—just the quotes themselves, each on a new line. Each quote must be self-contained, under 240 characters, and require no additional explanation."""},
                {"role": "user", "content": f"""Extract only the notable quotes from this segment. Return only the quotes, each on a new line. No summary, heading, or commentary.
{chunk}"""}
            ])
            
            if response:
                raw_quotes = response['choices'][0]['message']['content'].strip().split('\n')
                for quote in raw_quotes:
                    quote = quote.strip()
                    if not quote:
                        continue
                    
                    # Clean and format quote
                    quote = clean_quote(quote)
                    if len(quote) <= 240:  # Twitter limit
                        all_quotes.append(quote)
                        print(f"✅ Added quote ({len(quote)} chars)")
            
            # Add a small delay between chunks to avoid rate limits
            if i < len(chunks):
                time.sleep(1)
        
        print(f"\n✨ Generated {len(all_quotes)} total quotes")
        return all_quotes
        
    except Exception as e:
        print(f"Error extracting quotes: {str(e)}")
        return []

def extract_general_quotes(transcript: str) -> List[str]:
    """Extract general quotes from transcript without speaker information."""
    try:
        all_quotes = []
        chunks = chunk_transcript(transcript)
        
        logger.info(f"Starting general quote extraction with {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            if not chunk.strip():
                logger.warning(f"Chunk {i} is empty, skipping")
                continue
                
            # Ask DeepSeek to extract quotes from this chunk
            logger.info(f"Calling DeepSeek API for chunk {i}")
            response = call_deepseek_api([
                {"role": "system", "content": """You are an expert at identifying impactful and quotable moments from conversations. Your task is to extract the most interesting, insightful, or memorable quotes that would work well on social media. Each quote should:
1. Be self-contained and meaningful on its own
2. Be around 240 characters
3. Capture a key insight, opinion, or memorable statement
4. Represent a complete thought or important point

Format each quote on a new line.
Do not include any summary, heading, or commentary—just the quotes themselves, each on a new line. Each quote must be self-contained, under 240 characters, and require no additional explanation."""},
                {"role": "user", "content": f"""Extract the most notable and quotable statements from this text. Return only the quotes, each on a new line. No summary, heading, or commentary.
{chunk}"""}
            ])
            
            if not response:
                logger.error(f"No response from API for chunk {i}")
                continue
                
            if 'choices' not in response or not response['choices']:
                logger.error(f"Invalid API response format for chunk {i}: {response}")
                continue
                
            try:
                raw_quotes = response['choices'][0]['message']['content'].strip().split('\n')
                logger.info(f"Received {len(raw_quotes)} raw quotes from chunk {i}")
                
                for quote in raw_quotes:
                    quote = quote.strip()
                    if not quote:
                        continue
                    
                    # Clean and format quote
                    quote = clean_quote(quote)
                    if len(quote) <= 240:  # Twitter limit
                        all_quotes.append(quote)
                        logger.info(f"Added quote ({len(quote)} chars): {quote[:50]}...")
                    else:
                        logger.warning(f"Quote too long ({len(quote)} chars), skipping: {quote[:50]}...")
                
            except (KeyError, AttributeError, IndexError) as e:
                logger.error(f"Error processing API response for chunk {i}: {str(e)}")
                continue
            
            # Add a small delay between chunks to avoid rate limits
            if i < len(chunks):
                time.sleep(1)
        
        if not all_quotes:
            logger.warning("No quotes were generated from any chunks")
        else:
            logger.info(f"Successfully generated {len(all_quotes)} total quotes")
            
        return all_quotes
        
    except Exception as e:
        logger.exception(f"Error extracting general quotes: {str(e)}")
        return []

def create_quote_thread(transcript: str, metadata: Dict[str, Any], progress_callback: Optional[Dict[str, Any]] = None) -> List[str]:
    """Create a thread of quotes from the transcript."""
    try:
        # Initialize progress and logging
        logger.info("Starting quote thread creation")
        logger.info(f"Transcript length: {len(transcript)} chars")
        logger.info(f"Metadata: {metadata}")
        
        if progress_callback:
            progress_callback.update({
                'status': 'Starting quote generation...',
                'progress': 0.0
            })
        
        # Validate input
        if not transcript or not transcript.strip():
            error_msg = "Empty transcript provided"
            logger.error(error_msg)
            if progress_callback:
                progress_callback.update({
                    'status': error_msg,
                    'progress': 0.0
                })
            return []
            
        # If no speaker info, extract general quotes
        if not metadata.get('speaker'):
            logger.info("No speaker information provided, extracting general quotes...")
            response = call_deepseek_api([
                {"role": "system", "content": """You are an expert at extracting meaningful quotes from text. Focus on finding impactful, standalone statements that capture key ideas, insights, or memorable moments. Each quote should be self-contained and meaningful on its own.

For each quote:
- Keep it concise (1-3 sentences max)
- Ensure it can stand alone without context
- Preserve the original meaning and intent
- Focus on insights, key points, or memorable statements
- Avoid repetitive or redundant quotes
- Format each quote on its own line

Return only the quotes, one per line. Do not include any other text or formatting."""},
                {"role": "user", "content": f"Extract 3-5 meaningful quotes from this text:\n\n{transcript}"}
            ])
            
            if not response:
                error_msg = "No response from API for initial quote extraction"
                logger.error(error_msg)
                if progress_callback:
                    progress_callback.update({
                        'status': error_msg,
                        'progress': 0.0
                    })
                return []
            
            quotes = extract_quotes_from_response(response)
            if not quotes:
                error_msg = "No valid quotes found after processing"
                logger.error(error_msg)
                if progress_callback:
                    progress_callback.update({
                        'status': error_msg,
                        'progress': 0.0
                    })
                return []
                
            logger.info(f"Generated {len(quotes)} general quotes")
            return quotes

        # Process speaker-specific quotes
        speaker = metadata.get('speaker')
        logger.info(f"Processing speaker-specific quotes for {speaker}")
        
        # Parse transcript into segments
        all_speakers = []
        if metadata.get('space_info'):
            if metadata['space_info'].get('host'):
                all_speakers.append(metadata['space_info']['host'])
            if metadata['space_info'].get('speakers'):
                all_speakers.extend(metadata['space_info']['speakers'])
        
        if not all_speakers:
            logger.info("No speaker information available, falling back to general quotes")
            return extract_general_quotes(transcript)
            
        segments = parse_transcript_segments(transcript, all_speakers)
        if not segments:
            logger.warning(f"No segments found for {speaker}, falling back to general quotes")
            return extract_general_quotes(transcript)
            
        # Filter segments for this speaker
        speaker_segments = [s for s in segments if s['speaker'].lower() == speaker.lower()]
        if not speaker_segments:
            logger.warning(f"No segments found for speaker {speaker}, falling back to general quotes")
            return extract_general_quotes(transcript)
            
        # Process segments in chunks
        all_quotes = []
        speaker_text = '\n\n'.join([s['text'] for s in speaker_segments])
        chunks = chunk_transcript(speaker_text)
        
        logger.info(f"Processing {len(chunks)} chunks for speaker {speaker}")
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
            
            if not chunk.strip():
                logger.warning(f"Empty chunk {i}, skipping")
                continue
            
            # Update progress if callback provided
            if progress_callback:
                progress_callback.update({
                    'status': f'Processing chunk {i}/{len(chunks)}...',
                    'progress': i / len(chunks)
                })
            
            # Extract quotes from chunk
            response = call_deepseek_api([
                {"role": "system", "content": """You are an expert at identifying impactful and quotable moments from conversations. Your task is to extract the most interesting, insightful, or memorable quotes that would work well on Twitter. Each quote should:
1. Be self-contained and meaningful on its own
2. Be around 240 characters
3. Capture a key insight, opinion, or memorable statement
4. Be something worth quoting

Format each quote on a new line.
Do not include any summary, heading, or commentary—just the quotes themselves, each on a new line. Each quote must be self-contained, under 240 characters, and require no additional explanation."""},
                {"role": "user", "content": f"""Extract only the notable quotes from this segment. Return only the quotes, each on a new line. No summary, heading, or commentary.
{chunk}"""}
            ])
            
            if not response:
                logger.error(f"No response from API for chunk {i}")
                continue
            
            chunk_quotes = extract_quotes_from_response(response, i)
            all_quotes.extend(chunk_quotes)
            
            # Add a small delay between chunks to avoid rate limits
            if i < len(chunks):
                time.sleep(1)
        
        if not all_quotes:
            error_msg = "No quotes were generated from any chunks"
            logger.error(error_msg)
            if progress_callback:
                progress_callback.update({
                    'status': error_msg,
                    'progress': 0.0
                })
            return []
            
        logger.info(f"Successfully generated {len(all_quotes)} total quotes")
        return all_quotes
        
    except Exception as e:
        error_msg = f"Error generating quotes: {str(e)}"
        logger.exception(error_msg)
        if progress_callback:
            progress_callback.update({
                'status': error_msg,
                'progress': 0.0
            })
        return []

def format_quote_tweet(quote: str, speaker: str, context: str = '', speaker_info: Optional[Dict[str, Any]] = None) -> str:
    """Format a quote for Twitter."""
    # Clean the quote
    quote = clean_quote(quote)
    
    # Add speaker attribution if not already present
    if not quote.startswith(f"@{speaker}"):
        quote = f"@{speaker}: {quote}"
    
    return quote

def format_quotes(quotes: List[str], metadata: Dict[str, Any]) -> List[str]:
    """Format a list of quotes with proper attribution and context."""
    formatted = []
    
    # Get speaker info from metadata
    space_info = metadata.get('space_info', {})
    speaker = space_info.get('host', '')  # Default to host if no specific speaker
    
    for quote in quotes:
        if quote:
            formatted_quote = format_quote_tweet(quote, speaker, speaker_info=space_info)
            if formatted_quote and len(formatted_quote) <= 240:  # Twitter character limit
                formatted.append(formatted_quote)
    
    return formatted

def main():
    """Process a transcript and create a quote thread"""
    import argparse
    parser = argparse.ArgumentParser(description='Create a tweet thread of quotes from a Space transcript')
    parser.add_argument('transcript_path', help='Path to the transcript file')
    parser.add_argument('space_info_path', help='Path to the space info JSON file')
    args = parser.parse_args()
    
    # Load space info
    with open(args.space_info_path, 'r') as f:
        space_info = json.load(f)
    
    # Create quote thread
    tweets = create_quote_thread(open(args.transcript_path, 'r', encoding='utf-8').read(), space_info)
    
    # Print preview
    print("\n🎯 Quote Thread Preview:")
    for i, tweet in enumerate(tweets, 1):
        print(f"\nTweet {i}/{len(tweets)}")
        print("-" * 40)
        print(tweet)
        print("-" * 40)

if __name__ == "__main__":
    main()
