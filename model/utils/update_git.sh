#!/bin/bash

# Change directory to the script's folder
cd "$(dirname "$0")"

# Execute the rest of the commands
git pull

# Install packages directly into the py312 environment
#conda install --yes --name py312 -c conda-forge --file requirements.txt

source ~/anaconda3/bin/activate ~/anaconda3/envs/py312
pip install -r requirements.txt
