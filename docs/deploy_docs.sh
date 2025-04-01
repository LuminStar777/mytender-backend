#!/bin/bash
set -e

# Set working directory to the docs directory
cd "$(dirname "$0")"

# Check if documentation exists
if [ ! -d "build/html" ]; then
  echo "Documentation not found. Building it first..."
  ./build_docs.sh
fi

# Parse command line arguments
SERVER="44.208.84.199"
USER="ec2-user"
TARGET_DIR="/home/ec2-user/sphinx_doc_html"

# Ensure SSH key is available
SSH_KEY="$HOME/.ssh/id_rsa"
if [ ! -f "$SSH_KEY" ]; then
  echo "SSH key not found at $SSH_KEY"
  echo "Please ensure your SSH key is properly set up for deployment"
  exit 1
fi

echo "Preparing to deploy documentation to $SERVER:$TARGET_DIR"

# Create directory if it doesn't exist
ssh -i "$SSH_KEY" $USER@$SERVER "mkdir -p $TARGET_DIR"

# Clean target directory before deployment
echo "Cleaning target directory to remove old files..."
ssh -i "$SSH_KEY" $USER@$SERVER "rm -rf $TARGET_DIR/*"

# Copy documentation files
echo "Copying new documentation files..."
scp -i "$SSH_KEY" -r build/html/* $USER@$SERVER:$TARGET_DIR

echo "====================================="
echo "Documentation successfully deployed!"
echo "Available at: http://$SERVER/$TARGET_DIR/"
echo "=====================================" 