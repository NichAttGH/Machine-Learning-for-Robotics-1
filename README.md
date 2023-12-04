# Machine Learning 1
The assignments are:
1) Naive Bayes classifier
2) Linear regression
3) kNN Classifier
4) ?

For each Assignment, there is a report into each folder that explains what it consists of.

## 3) kNN classifier
The tasks of the III Assignment are:
- Task 1: Obtain a dataset
- Task 2: Build a kNN classifier
- Task 3: Test the kNN classifiers

### Introduction
The goal of this work is to create a k-nearest neighbors classifier in MATLAB.
The k-nearest neighbors algorithm (k-NN) is a non-parametric classification method.
It doesn’t explicitly implement a model with parameters but directly build a discrimination rule from data.
The data set used as training and test set is the *mnist* dataset (To download it, click here [mnist dataset](https://2023.aulaweb.unige.it/pluginfile.php/211469/mod_assign/intro/mnist.zip)).
Its data represent handwritten digits in 28 by 28 grey scale images.
Those are split up in 10 classes that represent the numbers from 0 to 9.
Each class has at least 5800 occurrences in the training set and 890 occurrences in the test set.

### Methods

#### Data Elaboration
Since the great number of occurrences for each class of the data set, it’s necessary to reduce what will be the training set.
To do that, a specified percentage of the whole set is taken.
In the data set the different classes are distributed consecutively.
To be sure all the classes are represented equally in the subset, it’s necessary to take the observations randomly.
Another way could be to create each class subset one by one, and then fuse the 10 subsets to create the training set.

#### kNN classifier
The Nearest Neighbor rule is based on the assumption that observations which
are close together (in some appropriate metric) will have the same classification ([Study of T. Cover and P. Hart](https://doi.org/10.1109/TIT.1967.1053964)).
In other words, given a training set:

![Training Dataset](assignment3/kNN_1.png)

![kNN classifier](assignment3/kNN_2.png)

#### The Matlab Function
The classifier function has at least 4 input arguments:
- The **training set**, a *n x d* matrix where *n* is the number of observations and *d* is the number of attributes.
  In the studied case each observation has 784 (28x28) attributes: each image, sampled as grey scale, is represented as a row vector of 784 numbers;
- The **training set labels**, a *n x 1* matrix with each value equal to the number the corresponding observation represents.
  Note that the "0" class is represented by the number *10*, in order to use it as index;
- The **test set**, a *m x d* matrix where *m* is the number of observations to classify and *d* remain the number of attribute.
  One of the firsts operations the function does, is to check if the *d-s* are the same;
- The **k number of neighbors**. This could be either a scalar or a vector of *k-s*.
  The function will return as many estimated classes as the number of *k-s* in the vector.

An additional input might be the **test set classes**. This, if present, is used to compute the accuracy once the function classify all the observations.
The program executes a number of checks before it starts the classification:
1. Checks that the number of arguments are at least 4;
2. Checks that the number of attributes of training and test set matches;
3. Checks that all the values in the *k* vector are greater than 0 but smaller than the total number of observations (The function cannot find 10 nearest
neighbors if there exist only 5 observations).

The main operation the function does is to call another function, the [*pdist2* function](https://it.mathworks.com/help/stats/pdist2.html): this returns the euclidean (by default) distance between the two
given observations.
When two matrices (training and test set) are given in input, it returns a matrix with the distances of each observation in the first matrix from each observation in the second one.

An optional input argument it has is the ’Smallest’ string followed by an integer *n*: for each observation in the test matrix, *pdist2* finds the *n* smallest
distances by computing and comparing the distance values to all the observations in the training matrix.
The function then sorts the distances of every test observation in ascending order.
In this case, the second output contains the indices of the observations in the training set corresponding to the distances just computed.

The configured function gives an *n* equal to the greater *k* in the homonym vector since no other neighbors are needed. For each *k* and for each test observation, the classifier computes the class label as the mode on the first *k* class labels taken with the previously obtained indices.
The output will be a matrix with as many rows as the test set observations and as many columns as the number of values in the *k* vector: for each observation in the test set, the function returns the most probable class for each *k*.

### Results
The test has been carried out using the whole training set, 50% of the test set and with these *k* values:

*k = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]*

Using a for loop, the function is called to classify the test set observation as belonging to the i<sub>th</sub> class or to any of the others.
Since the test set label are also given as input, the function is able to compute the accuracy in recognizing each digit versus the remaining, with a *k* value that ranges in the given vector.

![Results](assignment3/Results.png)

For *k = 3*, with an accuracy that ranged from 99.04 and 99.7, there were the biggest average accuracy of about 99.42.
The absolute best accuracy was obtained when recognizing the digit 0 with *k = 2*: the accuracy was of about 99.84.

In updating..
