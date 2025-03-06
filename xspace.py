import os
import sys
import argparse
import json
import time
from datetime import datetime
import yt_dlp
import re
from dotenv import load_dotenv
import whisper
import requests
from typing import Dict, List, Set, Optional, Tuple
from xspace_info import get_space_info
from xsummary import create_summary
from xthread import create_thread
from xquotes import create_quote_thread
import tweepy
import subprocess
import torch
from yt_dlp import YoutubeDL
import signal
from urllib.parse import urlparse
from tqdm import tqdm

# Load and validate environment
load_dotenv()
try:
    from validate_env import main as validate_env
    if validate_env() != 0:
        print("\n Environment validation failed. Please fix the issues above.")
        sys.exit(1)
except ImportError:
    print("\n Warning: validate_env.py not found. Skipping environment validation.")

def call_deepseek_api(messages, max_tokens=1000, temperature=0.7):
    """Make a call to the DeepSeek API"""
    headers = {
        'Authorization': f'Bearer {os.getenv("DEEPSEEK_API_KEY")}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'deepseek-chat',  # or whichever model you want to use
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    
    try:
        response = requests.post(os.getenv("DEEPSEEK_API_URL"), headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {str(e)}")
        if response := getattr(e, 'response', None):
            print(f"Response: {response.text}")
        raise

def clean_filename(title):
    """Clean filename by removing special characters and limiting length"""
    # Remove special characters and spaces, keep only alphanumeric and underscores
    import re
    clean = re.sub(r'[^\w\s-]', '', title)
    clean = re.sub(r'[-\s]+', '_', clean)
    # Take first 10 characters
    return clean[:10]

def format_output_filename(title: str, output_type: str) -> str:
    """Format output filename according to requirements"""
    # Take first 10 chars of title and clean it
    clean_title = clean_filename(title)[:10].strip()
    # Format date as MMDDHHMM
    date_str = datetime.now().strftime("%m%d%H%M")
    return f"{clean_title}_{date_str}_{output_type}.txt"

def get_project_dir(base_dir: str, title: str) -> str:
    """Create and return project-specific directory based on title"""
    # Take first 10 chars of title and clean it
    project_name = clean_filename(title)[:10].strip()
    project_dir = os.path.join(base_dir, project_name)
    
    # Create directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)
    
    return project_dir

def download_audio(url: str) -> Optional[str]:
    """Download audio from a Twitter/X Space using yt-dlp"""
    try:
        print("\n Downloading audio...")
        output_template = "downloads/%(id)s_%(upload_date)s.%(ext)s"
        
        ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
            
    except Exception as e:
        print(f"Error downloading audio: {str(e)}")
        return None

def transcribe_audio(audio_path: str, model_name: str = "base") -> Optional[Dict]:
    """Transcribe audio using Whisper"""
    try:
        print(f"\n Loading Whisper model '{model_name}'...")
        with tqdm(total=1, desc="Loading model", bar_format='{desc}: {percentage:3.1f}%|{bar}| {elapsed}') as pbar:
            model = whisper.load_model(model_name)
            pbar.update(1)
        
        print("\n Transcribing audio...")
        result = model.transcribe(audio_path)
        
        if not result or not result.get("text"):
            return None
            
        return result
        
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        return None

def normalize_space_info(space_info: Optional[Dict]) -> Dict:
    """Normalize space info to a flat structure for compatibility"""
    if not space_info:
        return {}
        
    normalized = {
        'title': space_info.get('title'),
        'url': space_info.get('url')
    }
    
    # Handle host info
    if host := space_info.get('host'):
        if isinstance(host, dict):
            normalized['host'] = host.get('handle')
            normalized['host_display_name'] = host.get('name')
        else:
            normalized['host'] = str(host)
    
    # Handle speakers info
    if speakers := space_info.get('speakers'):
        if speakers and isinstance(speakers[0], dict):
            normalized['speakers'] = [s.get('handle') for s in speakers]
            normalized['speaker_display_names'] = {
                s.get('handle'): s.get('name') 
                for s in speakers 
                if s.get('handle')
            }
        else:
            normalized['speakers'] = [str(s) for s in speakers]
    
    return normalized

def main():
    parser = argparse.ArgumentParser(description='Process Twitter/X Space audio')
    parser.add_argument('url', help='URL of the Twitter/X Space')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       default='base', help='Whisper model to use')
    args = parser.parse_args()

    # Get space info first
    print("\nGetting space information...")
    try:
        raw_space_info = get_space_info(args.url)
        space_info = normalize_space_info(raw_space_info)
        if not space_info:
            print(" Could not get space information. Continuing with limited metadata...")
            space_info = {
                'title': None,
                'url': args.url
            }
    except Exception as e:
        print(f" Error getting space information: {str(e)}")
        print("Continuing with limited metadata...")
        space_info = {
            'title': None,
            'url': args.url
        }

    # Download audio
    audio_path = download_audio(args.url)
    if not audio_path:
        return 1

    # Get project directory using space title if available
    title = space_info.get('title') or os.path.basename(audio_path)
    project_dir = get_project_dir(os.path.dirname(audio_path), title)

    # Transcribe audio
    result = transcribe_audio(audio_path, args.model)
    if not result:
        return 1

    transcript = result["text"].strip()

    # Create summary with space info
    print("\nGenerating summary...")
    summary = create_summary(transcript, space_info)
    if not summary:
        return 1

    # Create thread with space info
    print("\nGenerating thread...")
    thread = create_thread(summary, space_info)
    if not thread:
        return 1

    # Save transcript to project directory
    project_transcript_path = os.path.join(project_dir, format_output_filename(title, "transcript"))
    with open(project_transcript_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    print(f" Transcript saved to: {project_transcript_path}")

    # Save summary to project directory
    summary_path = os.path.join(project_dir, format_output_filename(title, "summary"))
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f" Summary saved to: {summary_path}")

    # Save thread to project directory
    thread_path = os.path.join(project_dir, format_output_filename(title, "thread"))
    with open(thread_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(thread))
    print(f" Thread saved to: {thread_path}")

    # Generate quotes with space info
    print("\n Generating quotes...")
    quotes = create_quote_thread(transcript, space_info)
    if quotes:
        quotes_path = os.path.join(project_dir, format_output_filename(title, "quotes"))
        with open(quotes_path, 'w', encoding='utf-8') as f:
            if isinstance(quotes, list):
                f.write('\n\n'.join(quotes))
            else:
                f.write(str(quotes))
        print(f" Quotes saved to: {quotes_path}")

    print("\n All done!")
    return 0

if __name__ == "__main__":
    main()