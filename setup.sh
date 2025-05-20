#!/usr/bin/env bash
set -e

# --------------------------------------------
# twitter-space-clipper setup script
# --------------------------------------------

echo "🔧 Setting up twitter-space-clipper..."

# 1️⃣  Detect whether 'uv' is available
if command -v uv >/dev/null 2>&1; then
	USE_UV=true
else
	USE_UV=false
fi

# 2️⃣  Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
	echo "📦 Creating virtual environment..."
	if [ "$USE_UV" = true ]; then
		uv venv
	else
		python3 -m venv .venv
	fi
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade build tool‑chain
echo "🔄 Upgrading pip, setuptools, wheel, and setuptools_scm..."
if [ "$USE_UV" = true ]; then
	uv pip install --upgrade pip setuptools wheel setuptools_scm
else
	pip install --upgrade pip setuptools wheel setuptools_scm
fi

# Install Python dependencies
echo "📥 Installing dependencies..."
if [ "$USE_UV" = true ]; then
	uv pip install -r requirements.txt
else
	pip install -r requirements.txt
fi

# Check for ffmpeg and give distro‑specific help
if ! command -v ffmpeg &>/dev/null; then
	echo "⚠️  ffmpeg not found."
	if [ -f /etc/arch-release ]; then
		echo "    💡 Arch Linux:     sudo pacman -S ffmpeg"
	elif grep -qi ubuntu /etc/os-release 2>/dev/null; then
		echo "    💡 Ubuntu/Debian:  sudo apt install ffmpeg"
	elif grep -qi fedora /etc/os-release 2>/dev/null; then
		echo "    💡 Fedora:         sudo dnf install ffmpeg"
	elif [ "$OS" = "Windows_NT" ]; then
		echo "    💡 Windows (Chocolatey):  choco install ffmpeg"
		echo "       or download a build from https://ffmpeg.org/"
	else
		echo "    💡 Please install ffmpeg using your system's package manager."
	fi
	exit 1
fi

echo "✅ Setup complete! Run './run.sh' to start the application."
