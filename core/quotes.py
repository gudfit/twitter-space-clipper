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

def extract_speaker_quotes(transcript: str, speaker_info: Dict) -> List[str]:
    """Extract the best quotes for each speaker using DeepSeek"""
    try:
        speaker = speaker_info.get('speaker')
        if not speaker:
            print(f"❌ No speaker provided")
            return []
            
        print(f"\n🔍 Analyzing transcript for quotes from {speaker}...")
        
        # Parse transcript into segments
        all_speakers = []
        if speaker_info.get('space_info'):
            if speaker_info['space_info'].get('host'):
                all_speakers.append(speaker_info['space_info']['host'])
            if speaker_info['space_info'].get('speakers'):
                all_speakers.extend(speaker_info['space_info']['speakers'])
        
        if not all_speakers:
            print("❌ No speaker information available")
            return []
            
        segments = parse_transcript_segments(transcript, all_speakers)
        if not segments:
            print(f"❌ No segments found for {speaker}")
            return []
            
        # Filter segments for this speaker
        speaker_segments = [s for s in segments if s['speaker'].lower() == speaker.lower()]
        if not speaker_segments:
            print(f"❌ No segments found for {speaker}")
            return []
            
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
4. Be something the speaker would be proud to have quoted

Format each quote on a new line, starting with the speaker's handle.
Do not include any summary, heading, or commentary—just the quotes themselves, each on a new line. Each quote must be self-contained, under 240 characters, and require no additional explanation."""},
                {"role": "user", "content": f"""Extract only the notable quotes from this segment by {speaker}. Return only the quotes, each on a new line. No summary, heading, or commentary.
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

def create_quote_thread(transcript: str, metadata: Dict[str, Any], progress_callback: Optional[Dict[str, Any]] = None) -> List[str]:
    """Create a thread of quotes from the transcript."""
    logger.info("Starting quote generation")
    
    try:
        # Split transcript into chunks
        chunks = chunk_transcript(transcript)
        total_chunks = len(chunks)
        
        if progress_callback:
            progress_callback.update({
                'status': f'Processing {total_chunks} chunks...',
                'progress': 0.1
            })
        
        quotes = []
        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress = 0.1 + (0.8 * (i / total_chunks))  # Scale from 10% to 90%
                progress_callback.update({
                    'status': f'Generating quotes from chunk {i+1}/{total_chunks}',
                    'progress': progress
                })
            
            # Generate quotes for this chunk
            chunk_quotes = extract_speaker_quotes(chunk, metadata)
            quotes.extend(chunk_quotes)
            
            # Log progress
            logger.info(f"Generated {len(chunk_quotes)} quotes from chunk {i+1}/{total_chunks}")
            
            if progress_callback:
                progress_callback.update({
                    'status': f'Generated {len(chunk_quotes)} quotes from chunk {i+1}/{total_chunks}',
                    'progress': progress
                })
        
        # Filter and format quotes
        if progress_callback:
            progress_callback.update({
                'status': 'Filtering and formatting quotes...',
                'progress': 0.9
            })
        
        formatted_quotes = format_quotes(quotes, metadata)
        
        if progress_callback:
            progress_callback.update({
                'status': f'Quote generation complete. Generated {len(formatted_quotes)} quotes.',
                'progress': 1.0
            })
        
        return formatted_quotes
        
    except Exception as e:
        logger.error(f"Error generating quotes: {str(e)}")
        if progress_callback:
            progress_callback.update({
                'status': f'Error: {str(e)}',
                'progress': 0.0
            })
        raise

def format_quote_tweet(quote: str, speaker: str, context: str = '', speaker_info: Optional[Dict[str, Any]] = None) -> str:
    """Format a quote as a tweet with proper attribution and context"""
    # Get speaker's role or affiliation if available
    role = speaker_info.get('roles', {}).get(speaker, '') if speaker_info else ''
    
    # Choose appropriate quote emoji based on context
    quote_emoji = '💭'
    if any(word in context.lower() for word in ['announce', 'reveal', 'launch']):
        quote_emoji = '📢'
    elif any(word in context.lower() for word in ['future', 'predict', 'vision']):
        quote_emoji = '🔮'
    elif any(word in context.lower() for word in ['tip', 'advice', 'recommend']):
        quote_emoji = '💡'
    
    # Format the tweet
    tweet = f'{quote_emoji} "{quote}"'
    
    # Add attribution
    attribution = f"- @{speaker}"
    if role:
        attribution += f" ({role})"
    
    # Combine while ensuring we're under character limit
    full_tweet = f"{tweet}\n\n{attribution}"
    if len(full_tweet) + 3 <= 280 and context:  # Add context if there's room
        full_tweet += f"\n\n📍 {context}"
    
    return full_tweet

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
