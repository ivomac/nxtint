#!/usr/bin/bash

# Exit on error
set -e

echo "Creating virtual environment..."
python -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt
pip install -e '.[dev]'

echo "Setting up environment variables..."
cp .env_template .env

echo "Installation complete!"
echo "Activate the virtual environment with: source .venv/bin/activate"
