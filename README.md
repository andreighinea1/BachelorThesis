# EEG Emotion Recognition Setup

This guide will help you set up the environment for EEG Emotion Recognition using Miniconda, FFMPEG, cuDNN, and PyTorch.

## 1. Clone the Repository

First, clone the repository to your local machine using Git.
```sh
git clone https://github.com/andreighinea1/BachelorThesis.git
cd BachelorThesis
```

## 2. Install Miniconda

Follow the instructions on the official [Miniconda installation page](https://docs.anaconda.com/miniconda/miniconda-install/) for your operating system.

### Windows
Download the Miniconda installer for Windows and run it. Follow the installation instructions provided on the page.

### Linux
Download the Miniconda installer for Linux and run the following commands in your terminal:
```sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts to complete the installation.

## 3. Install FFMPEG

Visit the [FFmpeg download page](https://ffmpeg.org/download.html) to get the latest release.

### Windows
1. Download the release for Windows.
2. Extract the downloaded file.
3. Add the `bin` directory to your system's PATH environment variable:
   - Open the Start Search, type in "env", and select "Edit the system environment variables".
   - Click on the "Environment Variables" button.
   - Under "System variables", find the `Path` variable and click "Edit".
   - Click "New" and add the path to the `bin` directory of your extracted FFMPEG folder.
   - Click "OK" to save and exit.

### Linux
1. Download the release for Linux.
2. Extract the downloaded file.
3. Add the `bin` directory to your system's PATH environment variable:
    ```sh
    export PATH=/path/to/ffmpeg/bin:$PATH
    ```
4. To make this change permanent, add the above line to your `.bashrc` or `.zshrc` file.

## 4. Install cuDNN

Follow the installation instructions from the official [cuDNN installation guide](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/overview.html) for your operating system.

### Windows
Refer to the [cuDNN installation guide for Windows](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html).

### Linux
Refer to the [cuDNN installation guide for Linux](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html).

## 5. Create and Activate Conda Environment

### Create Conda Environment with Python 3.11
```sh
conda create --name eeg_env python=3.11
```

### Activate the Environment

#### Windows
```sh
conda activate eeg_env
```

#### Linux
```sh
source activate eeg_env
```

## 6. Install PyTorch

Visit the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).

1. Select the following options:
   - **PyTorch Build**: Stable (2.3.1)
   - **Your OS**: Choose your operating system
   - **Package**: Conda
   - **Language**: Python
   - **Compute Platform**: CUDA 12.1

2. Copy the provided installation command and run it in your terminal. For example:
    ```sh
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

## 7. Install Other Packages

Ensure you are in the activated conda environment and install the other required packages from `requirements.txt`:
```sh
pip install -r requirements.txt
```

You have now set up your environment for EEG Emotion Recognition. If you encounter any issues, refer to the official documentation of the respective tools or libraries.