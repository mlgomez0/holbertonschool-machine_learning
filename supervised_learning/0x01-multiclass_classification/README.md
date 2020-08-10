# 0x01. Multiclass Classification

## Concepts

- What is multiclass classification?
- What is a one-hot vector?
- How to encode/decode one-hot vectors
- What is the softmax function and when do you use it?
- What is cross-entropy loss?
- What is pickling in Python?

# Installation
Files were interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Files were executed with numpy (version 1.15) and matplotlib (version 3.0)

## Usage

Educational purposes

## Tasks
0. One-Hot Encode: 0-one_hot_encode.py
```
Write a function def one_hot_encode(Y, classes): that converts a numeric label vector into a one-hot matrix:

Y is a numpy.ndarray with shape (m,) containing numeric class labels
m is the number of examples
classes is the maximum number of classes found in Y
Returns: a one-hot encoding of Y with shape (classes, m), or None on failure
```
1. One-Hot Decode: 1-one_hot_decode.py
```
Write a function def one_hot_decode(one_hot): that converts a one-hot matrix into a vector of labels:

one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
classes is the maximum number of classes
m is the number of examples
Returns: a numpy.ndarray with shape (m, ) containing the numeric labels for each example, or None on failure
```
2. Persistence is Key: 2-deep_neural_network.py
```
Update the class DeepNeuralNetwork (based on 23-deep_neural_network.py):

Create the instance method def save(self, filename):

Saves the instance object to a file in pickle format
filename is the file to which the object should be saved
If filename does not have the extension .pkl, add it
Create the static method def load(filename):

Loads a pickled DeepNeuralNetwork object
filename is the file from which the object should be loaded
Returns: the loaded object, or None if filename doesnt exist
```
3. Update DeepNeuralNetwork: 3-deep_neural_network.py
```
Update the class DeepNeuralNetwork to perform multiclass classification (based on 2-deep_neural_network.py):

You will need to update the instance methods forward_prop, cost, and evaluate
Y is now a one-hot numpy.ndarray of shape (classes, m)
Ideally, you should not have to change the __init__, gradient_descent, or train instance methods

Because the training process takes such a long time, I have pretrained a model for you to load and finish training (3-saved.pkl). This model has already been trained for 2000 iterations.
```
4. All the Activations: 4-deep_neural_network.py
```
Update the class DeepNeuralNetwork to allow different activation functions (based on 3-deep_neural_network.py):

Update the __init__ method to def __init__(self, nx, layers, activation='sig'):
activation represents the type of activation function used in the hidden layers
sig represents a sigmoid activation
tanh represents a tanh activation
if activation is not sig or tanh, raise a ValueError with the exception: activation must be 'sig' or 'tanh'
Create the private attribute __activation and set it to the value of activation
Create a getter for the private attribute __activation
Update the forward_prop and gradient_descent instance methods to use the __activation function in the hidden layers
```
