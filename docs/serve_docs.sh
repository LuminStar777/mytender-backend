#!/bin/bash
set -e

# Set working directory to the docs directory
cd "$(dirname "$0")"

# Check if the documentation exists
if [ ! -d "build/html" ]; then
    echo "Documentation not found. Building it first..."
    ./build_docs.sh
fi

echo "Starting HTTP server for documentation..."
echo "Documentation will be available at: http://localhost:8000/"
echo "Press Ctrl+C to stop the server"
echo ""

# Start a simple HTTP server in the background
cd build/html
python -m http.server 8000 