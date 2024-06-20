# Project Overview

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch.


# Architecture

Two different CNN architectures were implemented and compared:

## First Architecture:
Two convolutional layers followed by max-pooling layers.
Fully connected layers with dropout for regularization.
ReLU activation function for non-linearity.
Log Softmax for the output layer.

## Second Architecture:
Similar structure with adjusted number of filters and dropout rates to improve performance and prevent overfitting.

### First Architecture
- Conv Layer 1: 1 input channel, 10 output channels, kernel size 5
- Conv Layer 2: 10 input channels, 20 output channels, kernel size 5
- Dropout Layer: Dropout2d for regularization
- Fully Connected Layer 1: 320 input features, 50 output features
- Fully Connected Layer 2: 50 input features, 10 output features

### Second Architecture
- Conv Layer 1: 1 input channel, 32 output channels, kernel size 5
- Conv Layer 2: 32 input channels, 64 output channels, kernel size 5
- Dropout Layer 1: Dropout2d with p=0.2 for regularization
- Fully Connected Layer 1: 1024 input features, 128 output features
- Dropout Layer 2: Dropout with p=0.4 for regularization
- Fully Connected Layer 2: 128 input features, 10 output features


# Dataset
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is 28x28 pixels in grayscale.
"# handwritten-digits_mnist" 
