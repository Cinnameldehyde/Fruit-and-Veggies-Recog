Created a machine learning model using dataset from kaggle.
# Fruits and Veggies Recognition System

Using dataset: 

https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

https://www.kaggle.com/datasets/moltean/fruits

A CNN is a network that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image to differentiate one from the other.

To create our model, we’ll use Keras's sequential model.

There is a *MaxPooling2D* layer after every convolutional layer. This layer downsamples the input representation by taking the maximum value over a window. ‘Pooling’ is basically the process of merging for the purpose of reducing the size of the data.

*Flatten* converts the data into a 1-dimensional array for inputting it to the next layer. *Dropout* is a way to prevent overfitting in neural networks. The last layer has *‘softmax’* as the activation function.

The softmax activation function is **a mathematical function that converts a vector of real numbers into a probability distribution**. It exponentiates each element, making them positive, and then normalizes them by dividing by the sum of all exponentiated values.
That is, Softmax **assigns decimal probabilities to each class in a multi-class problem**. Those decimal probabilities must add up to 1.0. This additional constraint helps training converge more quickly than it otherwise would. Softmax is implemented through a neural network layer just before the output layer.

The number of *epochs* is a hyperparameter that defines the number of times that the learning algorithm will work through the entire training dataset.

The *batch_size* is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.

import numpy as np
import Tensorflow as tf
import matplotlib.pyplot as plt

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/f5f47f47-5d6c-4740-8338-2140f437b5bd/bddcf63c-eb15-4c37-9726-b55140e55552/Untitled.png)

https://kili-technology.com/training-data/training-validation-and-test-sets-how-to-split-machine-learning-data#

The model was trained on 100 epochs and has adequate accuracy rate as of my current level. 
The dataset consists of 36 different classes with each having 100 images of each kind. 
There are three phases: Training, Validation and Testing. 

This utitlizes different levels of input and output layers. The model utilizes CNN(Convolutional Neural Network). This is prepared as part of the Minor Project 
