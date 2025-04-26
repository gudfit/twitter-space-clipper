#!/bin/bash

# Exit on error
set -e

echo "🔧 Setting up twitter-space-clipper..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️ ffmpeg not found. Please install ffmpeg:"
    echo "Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "macOS: brew install ffmpeg"
    echo "Windows: Download from https://ffmpeg.org/download.html"
    exit 1
fi

echo "✅ Setup complete! Run './run.sh' to start the application." 