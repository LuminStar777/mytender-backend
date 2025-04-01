#!/bin/bash
set -e

# Navigate to the docs directory and run the build script
cd "$(dirname "$0")/docs"
./build_docs.sh 