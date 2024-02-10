#!/bin/bash

# Change directory to the script's folder
cd "$(dirname "$0")"

# Execute the rest of the commands
git pull

conda activate py312
conda install --yes --file requirements.txt
