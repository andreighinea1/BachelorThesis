# EEG Emotion Recognition Setup

This project is the source code for the bachelor thesis titled `EmotionWave: GAN-AUGMENTED CUSTOM DATASET USING EEG HEADSET FOR EMOTION RECOGNITION`.
You can find the repository on GitHub [here](https://github.com/andreighinea1/BachelorThesis).

This guide will help you set up the environment for EEG Emotion Recognition using Miniconda, FFMPEG, cuDNN, and PyTorch.

## Overview

This project focuses on EEG emotion recognition through advanced machine learning techniques.
It includes the following capabilities:

- **Dataset Collection**: Using prepared `.mp4` videos categorized by emotions (negative, neutral, positive).
- **Model Training and Evaluation**: Includes baseline training on the SEED dataset and custom dataset EmotionWave.
- **GAN Implementation**: Utilizes GANs for data augmentation.

All model code is located in the `model` directory, and paths mentioned will be relative to this directory.

### EmotionWave Dataset

For access to the EmotionWave dataset, you may contact the project owner at `andreidezvoltator@gmail.com` to request the dataset.

## Prerequisites

Before diving into the installation steps, ensure you have the following prerequisites:

- Miniconda
- FFMPEG
- cuDNN
- Git
- PyCharm Professional with Jupyter Notebook extension (VSCode is also possible but not covered here)

Instead of using Conda, you may also use a Python virtual environment, but the guide will focus on Conda.

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

## Project Capabilities

### Dataset Collection

1. **Prepare Videos**:
    - Place `.mp4` videos in `dataset_collection/videos/{EMOTION}` directories, with `{EMOTION}` being one of `negative, neutral, positive`. Name the videos as `1.mp4`, `2.mp4`, `3.mp4`, etc.

2. **Run Jupyter Notebook**:
    - Open and run `dataset_collection/notebook_dataset_collection.ipynb`.
    - This notebook concatenates the videos for each emotion and creates segments (choose segment length, default is 4 minutes).
    - Uncomment the line `# experiment.run_experiment()` to show the video and experiment setup.

### Model Training and Evaluation

- **SEED Dataset**:
    - Use `notebook.ipynb` to load the SEED Dataset, perform data augmentation, pre-training, fine-tuning, and evaluate the model.

- **Custom Dataset (EmotionWave)**:
    - Use `notebook_my_dataset_processing.ipynb` to load the EmotionWave dataset, run GANs for each channel, perform data augmentation, pre-training, fine-tuning, and evaluate the model.
    - Uncomment the line `# gan_manager.initialize_and_train_gans()` to train GANs. By default, GANs are loaded with `gan_manager.load_gan_models(epochs)`.

- **Hyperparameter Tuning**:
    - Use `notebook_testing_values.ipynb` for hyperparameter tuning to find the best values for training the model.

## Development Environment

For development, it's recommended to use PyCharm Professional with the Jupyter Notebook extension. If you prefer using VSCode, it is possible but not covered in this guide.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

You have now set up your environment and are ready to work on EEG Emotion Recognition. If you encounter any issues, refer to the official documentation of the respective tools or libraries.