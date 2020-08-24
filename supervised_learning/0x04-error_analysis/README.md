# 0x04. Error Analysis

## Concepts

- What is the confusion matrix?
- What is type I error? type II?
- What is sensitivity? specificity? precision? recall?
- What is an F1 score?
- What is bias? variance?
- What is irreducible error?
- What is Bayes error?
- How can you approximate Bayes error?
- How to calculate bias and variance
- How to create a confusion matrix

# Installation
Files were interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Files were executed with numpy (version 1.15) and matplotlib (version 3.0)
Your files will be executed with numpy (version 1.15) and tensorflow (version 1.12)

## Usage

Educational purposes

## Tasks

0. Create Confusion: 0-create_confusion.py
```
Write the function def create_confusion_matrix(labels, logits): that creates a confusion matrix:

labels is a one-hot numpy.ndarray of shape (m, classes) containing the correct labels for each data point
m is the number of data points
classes is the number of classes
logits is a one-hot numpy.ndarray of shape (m, classes) containing the predicted labels
Returns: a confusion numpy.ndarray of shape (classes, classes) with row indices representing the correct labels and column indices representing the predicted labels
```
1. Sensitivity: 1-sensitivity.py
```
Write the function def sensitivity(confusion): that calculates the sensitivity for each class in a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the sensitivity of each class
```
2. Precision: 2-precision.py
```
Write the function def precision(confusion): that calculates the precision for each class in a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the precision of each class
```
3. Specificity: 3-specificity.py
```
Write the function def specificity(confusion): that calculates the specificity for each class in a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the specificity of each class
```
4. F1 score: 4-f1_score.py
```
Write the function def f1_score(confusion): that calculates the F1 score of a confusion matrix:

confusion is a confusion numpy.ndarray of shape (classes, classes) where row indices represent the correct labels and column indices represent the predicted labels
classes is the number of classes
Returns: a numpy.ndarray of shape (classes,) containing the F1 score of each class
You may use sensitivity = __import__('1-sensitivity').sensitivity and precision = __import__('2-precision').precision
```
5. Dealing with Error: 5-error_handling
```
In the text file 5-error_handling, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. A,B,C):

Scenarios:

1. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance
Approaches:

A. Train more
B. Try a different architecture
C. Get more data
D. Build a deeper network
E. Use regularization
F. Nothing
```
