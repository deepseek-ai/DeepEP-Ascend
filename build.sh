#!/bin/bash
# Build a wheel in dist; this does not install or create an in-place extension.
set -euo pipefail

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py bdist_wheel

# Return to users' original directory
cd "$original_dir"
