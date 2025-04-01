#!/bin/bash
set -e

# Set working directory to the docs directory
cd "$(dirname "$0")"

echo "mytender.io Documentation Builder"
echo "====================================="

# Check if virtual environment exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Generate the RAG architecture diagram
echo "Generating RAG architecture diagram..."
python rag_architecture_diagram.py

# Build the documentation
echo "Building Sphinx documentation..."
sphinx-build -b html source build/html

# Print success message
echo "====================================="
echo "Documentation successfully built!"
echo "View the documentation by opening:"
echo "$(pwd)/build/html/index.html"
echo ""
echo "Or run: open build/html/index.html"

# Deactivate virtual environment
deactivate

exit 0 