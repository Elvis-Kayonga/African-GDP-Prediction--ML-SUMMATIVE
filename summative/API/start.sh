#!/bin/bash
# Startup script for Render

echo "Starting African GDP Prediction API..."
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# List installed packages
echo "Installed packages:"
pip list

# Check if model files exist
echo "Checking model files..."
ls -la *.pkl

# Start the application
echo "Starting uvicorn..."
uvicorn prediction:app --host 0.0.0.0 --port ${PORT:-8000}
