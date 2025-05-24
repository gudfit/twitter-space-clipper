"""Core functionality for transcribing audio files (now with timestamps)."""

import logging
import argparse
from typing import Optional, Dict, Any

import whisper  # type: ignore
import torch

logger = logging.getLogger(__name__)


def _fmt_ts(seconds: float) -> str:
    """Return a string MM:SS.mmm given seconds as float."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


def transcribe_audio(
    audio_path: str,
    progress_callback: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Transcribe *audio_path* with Whisper and **return a timestamped transcript**.

    Every line is formatted exactly like:
        [00:00.000 -> 00:07.120] Hello and welcome…

    which is the format other modules (clips, summary, etc.) already parse.
    """
    logger.info("Starting transcription of %s", audio_path)

    try:
        if progress_callback:
            progress_callback.update(
                {"status": "Loading audio file…", "progress": 0.05}
            )
        audio = whisper.load_audio(audio_path)

        if progress_callback:
            progress_callback.update(
                {"status": "Loading Whisper model…", "progress": 0.10}
            )
        model = whisper.load_model("base")

        if progress_callback:
            progress_callback.update(
                {"status": "Detecting language…", "progress": 0.15}
            )
        audio_features = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio_features).to(model.device)
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        logger.info("Detected language: %s", detected_language)

        if progress_callback:
            progress_callback.update(
                {
                    "status": f"Transcribing ({detected_language})…",
                    "progress": 0.20,
                }
            )
        result = model.transcribe(
            audio,
            language=detected_language,
            fp16=torch.cuda.is_available(),
            verbose=False,
        )

        if progress_callback:
            progress_callback.update(
                {"status": "Formatting transcript…", "progress": 0.90}
            )

        lines: list[str] = []
        for seg in result["segments"]:
            start, end, text = seg["start"], seg["end"], seg["text"].strip()
            lines.append(f"[{_fmt_ts(start)} -> {_fmt_ts(end)}] {text}")

        if progress_callback:
            progress_callback.update(
                {"status": "Transcription complete", "progress": 1.0}
            )

        return "\n".join(lines)

    except Exception as e:
        logger.error("Error transcribing audio: %s", e)
        if progress_callback:
            progress_callback.update({"status": f"Error: {e}", "progress": 0.0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe audio with Whisper and emit a timestamped transcript"
    )
    parser.add_argument("audio_path", help="Path to the audio file to transcribe")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: base)",
    )
    args = parser.parse_args()

    # Print the transcript to stdout when run directly
    transcript_text = transcribe_audio(args.audio_path)
    print(transcript_text)
