#!/bin/bash
# Development build: compile the C++ extension and symlink .so into the package
set -euo pipefail

# Change current directory into project root
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

# Remove old dist file, build files, and build
rm -rf build dist
rm -rf *.egg-info
python setup.py build

# Find the .so file in build directory and create symlink in current directory
so_file=$(find build -name "_C*.so" -type f -print -quit)
if [ -n "$so_file" ]; then
    ln -sf "../$so_file" deep_ep/
    echo "Success: linked deep_ep/$(basename "$so_file") -> ../$so_file"
else
    echo "Error: No SO file found in build directory" >&2
    exit 1
fi

# Return to users' original directory
cd "$original_dir"
