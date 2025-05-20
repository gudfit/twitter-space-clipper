#!/bin/bash
set -e

echo "🔧 Setting up twitter-space-clipper..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
	echo "📦 Creating virtual environment..."
	python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade build toolchain to avoid known packaging bugs
echo "🔄 Upgrading pip, setuptools, wheel, and setuptools_scm..."
pip install --upgrade pip setuptools wheel setuptools_scm

# Install Python dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for ffmpeg
if ! command -v ffmpeg &>/dev/null; then
	echo "⚠️ ffmpeg not found. On Arch Linux install with:"
	echo "    sudo pacman -S ffmpeg"
	exit 1
fi

echo "✅ Setup complete! Run './run.sh' to start the application."
