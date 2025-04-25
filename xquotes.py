import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = os.getenv('DEEPSEEK_API_URL')

def call_deepseek_api(messages: List[Dict]) -> Dict:
    """Call DeepSeek API with retry logic"""
    print("\n🔄 Preparing DeepSeek API call...")
    print(f"Model: deepseek-chat")
    print(f"Max tokens: 2000")
    print(f"Temperature: 0.7")
    print(f"Message count: {len(messages)}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        # Add /chat/completions to the API endpoint
        api_url = DEEPSEEK_API_URL.rstrip('/') + '/chat/completions'
        print(f"\n📡 Sending request to: {api_url}")
        print(f"System prompt length: {len(messages[0]['content'])}")
        print(f"User prompt length: {len(messages[1]['content'])}")
        
        response = requests.post(api_url, headers=headers, json=data)
        print(f"\n📥 Response received (status: {response.status_code})")
        
        response.raise_for_status()
        result = response.json()
        
        # Log token usage if available
        if 'usage' in result:
            usage = result['usage']
            print(f"\n📊 Token usage:")
            print(f"- Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"- Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"- Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        message = result['choices'][0]['message']
        print(f"\n✅ Successfully extracted response (length: {len(message['content'])} chars)")
        return message
        
    except Exception as e:
        print(f"\n❌ Error calling DeepSeek API: {str(e)}")
        if response := getattr(e, 'response', None):
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        return None

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
            
        # Combine segments into context for DeepSeek
        context = "\n\n".join([
            f"{s['speaker']}: {s['text']}"
            for s in speaker_segments
        ])
        
        # Ask DeepSeek to extract the best quotes
        response = call_deepseek_api([
            {"role": "system", "content": """You are an expert at identifying impactful and quotable moments from conversations. Your task is to extract the most interesting, insightful, or memorable quotes that would work well on Twitter. Each quote should:
1. Be self-contained and meaningful on its own
2. Be under 240 characters
3. Capture a key insight, opinion, or memorable statement
4. Be something the speaker would be proud to have quoted

Format each quote on a new line, starting with the speaker's handle."""},
            {"role": "user", "content": f"""Extract the best quotes from these segments by {speaker}. Format each quote on a new line starting with their handle.

{context}"""}
        ])
        
        if not response:
            return []
            
        quotes = []
        raw_quotes = response['choices'][0]['message']['content'].strip().split('\n')
        
        for quote in raw_quotes:
            quote = quote.strip()
            if not quote:
                continue
            
            # Clean and format quote
            quote = clean_quote(quote)
            if len(quote) <= 240:  # Twitter limit
                quotes.append(quote)
        
        return quotes
        
    except Exception as e:
        print(f"Error extracting quotes: {str(e)}")
        return []

def create_quote_thread(transcript: str, space_info: Dict) -> List[str]:
    """Create a thread of the best quotes from the Space"""
    print("\n🎯 Starting quote generation process...")
    print(f"Transcript length: {len(transcript)} characters")
    print(f"Space info: {space_info}")  # Log space info for debugging
    
    if not space_info or not transcript:
        print("❌ Missing required data (space_info or transcript)")
        return []
    
    # Extract quotes for each speaker
    quotes = []
    all_speakers = [space_info.get('host')] + space_info.get('speakers', []) if space_info.get('host') else space_info.get('speakers', [])
    
    if all_speakers:
        print(f"\n👥 Found {len(all_speakers)} speakers")
        for speaker in all_speakers:
            print(f"\n🎤 Processing speaker: {speaker}")
            if not speaker:
                print("⚠️ Skipping empty speaker name")
                continue
            speaker_quotes = extract_speaker_quotes(transcript, {'speaker': speaker, 'space_info': space_info})
            if speaker_quotes:
                print(f"✅ Generated {len(speaker_quotes)} quotes for {speaker}")
                quotes.extend(speaker_quotes)
            else:
                print(f"⚠️ No quotes generated for {speaker}")
    else:
        print("⚠️ No speakers found in space_info")
    
    # If no quotes found, try extracting without speaker attribution
    if not quotes:
        print("\n🔄 No speaker quotes found, trying general quote extraction...")
        response = call_deepseek_api([
            {"role": "system", "content": """You are an expert at identifying impactful and quotable moments from conversations. Extract the most interesting, insightful, or memorable quotes that would work well on Twitter. Each quote should:
1. Be self-contained and meaningful on its own
2. Be under 240 characters
3. Capture a key insight, opinion, or memorable statement"""},
            {"role": "user", "content": f"Extract the best quotes from this transcript:\n\n{transcript}"}
        ])
        
        if response and response.get('content'):
            print("\n📝 Processing DeepSeek response...")
            raw_quotes = response['content'].strip().split('\n')
            print(f"Found {len(raw_quotes)} raw quotes")
            
            for quote in raw_quotes:
                quote = clean_quote(quote)
                if quote and len(quote) <= 240:
                    quotes.append(quote)
                    print(f"✅ Added quote ({len(quote)} chars): {quote[:50]}...")
                else:
                    if not quote:
                        print("⚠️ Skipping empty quote")
                    else:
                        print(f"⚠️ Skipping quote - too long ({len(quote)} chars)")
            
            print(f"\n✨ Added {len(quotes)} valid quotes")
        else:
            print("❌ Failed to generate general quotes - invalid or empty response")
            print(f"Response: {response}")
    
    print(f"\n🎉 Quote generation complete! Generated {len(quotes)} total quotes")
    return quotes

def format_quote_tweet(quote: str, speaker: str, context: str = '', speaker_info: Dict = None) -> str:
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
