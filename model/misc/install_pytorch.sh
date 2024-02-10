#!/bin/bash

# Check if nvidia-smi command exists and is executable
if command -v nvidia-smi &> /dev/null
then
    # If nvidia-smi is available, attempt to run it and check for CUDA
    if nvidia-smi &> /dev/null
    then
        echo "CUDA detected, installing PyTorch with CUDA support."
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    else
        echo "nvidia-smi could not be executed, but it exists. Checking for a better way might be needed."
    fi
else
    echo "CUDA not detected, installing CPU-only PyTorch."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
fi
