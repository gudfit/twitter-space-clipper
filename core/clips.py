#!/usr/bin/env python3
import os
import sys
import re
import argparse
from pathlib import Path
import subprocess
import json
from typing import List, Tuple, Dict, Optional

def parse_timestamp_line(line: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse a timestamp line from the transcript"""
    try:
        # Format: [00:00.000 -> 00:30.000] Text here
        match = re.match(r'\[(\d+:\d+\.\d+)\s*->\s*(\d+:\d+\.\d+)\]\s*(.+)', line)
        if not match:
            return None, None, None
            
        start_str, end_str, text = match.groups()
        
        # Convert MM:SS.mmm to seconds
        def to_seconds(ts: str) -> float:
            mins, rest = ts.split(':')
            secs, ms = rest.split('.')
            return int(mins) * 60 + int(secs) + int(ms) / 1000
            
        return to_seconds(start_str), to_seconds(end_str), text.strip()
    except Exception:
        return None, None, None

def extract_quote_timestamps(quotes_file: str) -> List[Tuple[float, float, str]]:
    """Extract timestamps and text from quotes file"""
    transcript_file = quotes_file.replace('_quotes.txt', '_transcript.txt')
    if not os.path.exists(transcript_file):
        print(f"❌ Could not find transcript file: {transcript_file}")
        return []

    # Load transcript segments
    segments = []
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '[' in line and '->' in line:
                    start, end, text = parse_timestamp_line(line)
                    if start is not None:
                        segments.append((start, end, text))
    except Exception as e:
        print(f"Error reading transcript: {e}")
        return []

    if not segments:
        print("❌ No segments found in transcript")
        return []

    # Read quotes
    with open(quotes_file, 'r', encoding='utf-8') as f:
        quotes = f.readlines()

    # Extract quotes and find their timestamps
    quote_segments = []
    for quote in quotes:
        quote = quote.strip()
        if not quote:
            continue

        # Remove quote number and clean text
        clean_quote = re.sub(r'^\d+\.\s*"?', '', quote)
        clean_quote = clean_quote.strip('"')
        if '–' in clean_quote:  # Remove attribution
            clean_quote = clean_quote[:clean_quote.rindex('–')].strip()
        clean_quote = clean_quote.strip('"').strip()

        # Find matching segment in transcript
        for start, end, text in segments:
            if clean_quote.lower() in text.lower():
                quote_segments.append((start, end, quote))
                print(f"✅ Found timestamp for: {quote[:50]}...")
                break

    return quote_segments

def create_clip(input_file: str, output_file: str, start_time: float, end_time: float, quote: str) -> bool:
    """Create audio clip using ffmpeg"""
    duration = end_time - start_time
    try:
        # Add 0.5 second padding to start and end
        start_pad = max(0, start_time - 0.5)
        duration_pad = duration + 1.0  # Add total 1 second padding

        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-ss', str(start_pad),
            '-t', str(duration_pad),
            '-c', 'copy',
            '-y',  # Overwrite output file if it exists
            output_file
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ Created clip for quote: {quote[:50]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating clip: {e}")
        if e.stderr:
            print(f"ffmpeg error: {e.stderr.decode()}")
        return False

def process_quotes_file(quotes_file: str, audio_file: str, output_dir: str):
    """Process quotes file and create clips"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract timestamps from quotes
    quote_segments = extract_quote_timestamps(quotes_file)
    if not quote_segments:
        print("❌ No quotes with timestamps found")
        return
    
    # Process each quote
    for i, (start_time, end_time, quote) in enumerate(quote_segments, 1):
        # Create output filename
        output_file = os.path.join(output_dir, f'clip_{i:03d}.m4a')
        
        print(f"\nCreating clip {i}: {start_time:.2f}s to {end_time:.2f}s")
        if create_clip(audio_file, output_file, start_time, end_time, quote):
            print(f"📁 Saved as: {output_file}")
        else:
            print(f"❌ Failed to create clip {i}")

def find_audio_file(base_dir: str) -> Optional[str]:
    """Find the audio file in the given directory"""
    audio_extensions = ['.m4a', '.mp3', '.wav']
    for ext in audio_extensions:
        files = list(Path(base_dir).glob(f'*{ext}'))
        if files:
            return str(files[0])
    return None

def main():
    parser = argparse.ArgumentParser(description='Create audio clips from quotes')
    parser.add_argument('quotes_file', help='Path to the quotes file')
    parser.add_argument('--audio', help='Path to the audio file (optional)')
    parser.add_argument('--output', help='Output directory for clips (optional)')
    args = parser.parse_args()
    
    # Determine audio file path
    if args.audio:
        audio_file = args.audio
    else:
        # Look for audio file in the same directory as quotes file
        quotes_dir = os.path.dirname(args.quotes_file)
        audio_file = find_audio_file(quotes_dir)
        if not audio_file:
            print("❌ Could not find audio file. Please specify with --audio")
            sys.exit(1)
    
    # Determine output directory
    if args.output:
        output_dir = args.output
    else:
        # Create 'clips' directory next to quotes file
        output_dir = os.path.join(os.path.dirname(args.quotes_file), 'clips')
    
    print(f"📝 Quotes file: {args.quotes_file}")
    print(f"🎵 Audio file: {audio_file}")
    print(f"📂 Output directory: {output_dir}")
    
    process_quotes_file(args.quotes_file, audio_file, output_dir)

if __name__ == "__main__":
    main()
