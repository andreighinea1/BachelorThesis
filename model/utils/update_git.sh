#!/bin/bash

# Change directory to the script's folder
cd "$(dirname "$0")"

# Execute the rest of the commands
git pull

# Install packages directly into the py312 environment
conda install --yes --name py312 -c conda-forge --file requirements.txt
