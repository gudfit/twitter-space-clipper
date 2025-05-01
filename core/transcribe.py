"""Core functionality for transcribing audio files."""
import os
import logging
from typing import Optional, Dict, Any
import whisper  # type: ignore
import torch

# Configure module logger
logger = logging.getLogger(__name__)

def transcribe_audio(audio_path: str, progress_callback: Optional[Dict[str, Any]] = None) -> str:
    """Transcribe audio file using Whisper."""
    logger.info(f"Starting transcription of {audio_path}")
    
    try:
        # Load audio file
        if progress_callback:
            progress_callback.update({
                'status': 'Loading audio file...',
                'progress': 0.1
            })
        audio = whisper.load_audio(audio_path)
        
        # Load model
        if progress_callback:
            progress_callback.update({
                'status': 'Loading Whisper model...',
                'progress': 0.2
            })
        model = whisper.load_model("base")
        
        # Detect language
        if progress_callback:
            progress_callback.update({
                'status': 'Detecting language...',
                'progress': 0.3
            })
        audio_features = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_features).to(model.device)
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        logger.info(f"Detected language: {detected_language}")
        
        if progress_callback:
            progress_callback.update({
                'status': f'Transcribing audio (detected language: {detected_language})...',
                'progress': 0.4
            })
        
        # Transcribe
        result = model.transcribe(
            audio,
            language=detected_language,
            fp16=False,
            verbose=True
        )
        
        if progress_callback:
            progress_callback.update({
                'status': 'Transcription complete',
                'progress': 1.0
            })
        
        return result["text"]
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI's Whisper")
    parser.add_argument("audio_path", help="Path to the audio file to transcribe")
    parser.add_argument("--model", default="base", 
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use (default: base)")
    
    args = parser.parse_args()
    transcribe_audio(args.audio_path, args.model)
