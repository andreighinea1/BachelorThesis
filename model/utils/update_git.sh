#!/bin/bash

# Change directory to the script's folder
cd "$(dirname "$0")"

# Execute the rest of the commands
git pull

source ~/anaconda3/bin/activate ~/anaconda3/envs/py312
pip install -r requirements.txt
