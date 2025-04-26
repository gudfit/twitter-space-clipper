import whisper
import argparse

def transcribe_audio(audio_path, model_name="base"):
    """
    Transcribe audio using OpenAI's Whisper model
    
    Args:
        audio_path (str): Path to the audio file
        model_name (str): Name of the Whisper model to use 
                         (tiny, base, small, medium, large)
    """
    # Load the model
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    # Transcribe the audio
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    
    # Print the transcription
    print("\nTranscription:")
    print(result["text"])
    
    return result["text"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using OpenAI's Whisper")
    parser.add_argument("audio_path", help="Path to the audio file to transcribe")
    parser.add_argument("--model", default="base", 
                      choices=["tiny", "base", "small", "medium", "large"],
                      help="Whisper model to use (default: base)")
    
    args = parser.parse_args()
    transcribe_audio(args.audio_path, args.model)
