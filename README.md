# PPCA AI Battle 2022 (Week 3~4)

**In the AI Battle project, we will focus on learning about neural networks in Week 3~4.**

## Learning goals

- Learn to program in Python
- Understand the basic usage of Pytorch
- Explore how neural networks work
- Catch a glimpse of some classic neural network architectures
  - CNN
  - ResNet
  - Transformer
  - ...
- Train a neural network model for playing Gomoku 9x9

## Installation

Only the environment installation guide for ubuntu is provided.

### Miniconda installation

https://docs.conda.io/en/latest/miniconda.html

### Pytorch installation via conda

    conda create -n "your_venv_name" python=3.9
    conda activate your_venv_name

#### GPU

If you have a discrete graphics card, use the following installation instructions.

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

#### CPU only

    conda install pytorch torchvision torchaudio cpuonly -c pytorch

### Tensorboard installation via conda

    conda install tensorboardX


## Execution

### Training

Download the dataset for Gomoku 9x9 and unzip it into the directory of this repository.

https://jbox.sjtu.edu.cn/l/H1Ci9a

Start training a neural network model for Gomoku 9x9:

    python main.py

During the training process, the latest model will pit against a old version model and the random-player every 10 rounds.

The training will last for 50 iterations.

### Pitting

Play AI-AI or human-AI battles.

    python pit.py