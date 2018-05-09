
![network](https://github.com/godspeed1989/pntconv/blob/master/doc/teaser.png)

### Introduction
Code for ``.

### Installation

Install TensorFlow and h5py. The code has been tested with Python 3.6, TensorFlow 1.7.0, CUDA 9.1 and cuDNN 7.1 on ArchLinux.

### Usage

To train or evaluate or visualize:

    python run_cls.py

Log files and network parameters will be saved to `log_model_dataset` folder in default.

We can use TensorBoard to view the network architecture and monitor the training progress.

    tensorboard --logdir ./log_model_dataset


