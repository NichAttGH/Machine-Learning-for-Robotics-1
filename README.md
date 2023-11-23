# Machine Learning 1
The assignments are:
1) Naive Bayes classifier
2) Linear regression
3) kNN Classifier
4) ?

## 1) Naive Bayes classifier
The tasks of the I Assignment are:
- Task 1: Data preprocessing
- Task 2: Build a naive Bayes classifier
- Task 3: Improve the classifier with Laplace (additive) smoothing

### Introduction
The goal of this work is to develop a Naive Bayes Classifier using MATLAB.

### Methods
The Classifier will eventually be tested on a Weather data set.
The latter, as well as any other data set, needs some pre-processing in order to be manipulated from autonomous programs.
This particularly data set has some categorical features that needs to be converted to numeric form as soon as the raw data are loaded into the MATLAB workspace.
After all this representation conversions, the whole matrix data-type is changed, since the function used to load the data returns a cell matrix that is hard to work on.
This procedure is a lot quicker than a hand-done conversion, especially when it comes to data set with hundreds of thousands observations.
From the data set, the training and test set are randomly extrapolated.
This action is performed using a MATLAB function that returns random numbers used as indexes.

### Classification
Before the program starts the classification, it should check the given data are eligible to said classification.
First of all the test set’s number of features should match the training set’s one.
The number of columns could not match only when the training set has one more.
This happens when the test set doesn’t come with the target.
It’s also important the single units of data are consistent. Since the data set is preprocessed converting every possible level to a specific integer, no one of them should be lesser than one.
The program needs to know how many different classes it’s going to classify the observations into.
To do that, it simply count the number of unique values in the target column of the training set.
In order to compute the likelihood for each possible level of each attribute for every class, the software follows this algorithm: ![Likelihood P(x=v|c) estimation](assignment1/Algorithm_1.png)

Practically, the returned likelihood matrix is a cell matrix with as many rows as the number of classes and as many columns as the number of attribute.
Each value of the matrix is an array that stores the frequency of each possible level of the specific attribute in the specific class.
With this data, it’s possible to compute the overall probability that a given observation belong to a class rather than another.

![Discriminant function](assignment1/Algorithm_2.png)

Now it’s possible to compare the values of the discriminant function for each observation.
The probable belonging class is going to be the one with a greater value.
Once the program computes and stores the estimated target in a matrix, it’s finally possible to elaborate the error rate.
This action is obviously executed only if the test set target is given.
It’s equal to the number of incorrect classification divided by the number of observations.

### Laplace smoothing
It happens, especially when working on a small data set such as the one in object, that some combinations appear exclusively in the test set.
In this case, the program would’ve tried to retrieve the frequency of a value never encountered before.
So, a particular statement would’ve set its probability to zero.
This assignment, even if it avoids errors in the code, is not correct; in fact, even if a particular value it’s not present in the training set, doesn’t mean its probability is zero.
In order to correct this behavior, it’s given the possibility, when calling the classifier function, to use the *Laplace (Additive) Smoothing algorithm*. This
introduces a different way to compute the probability of observing a specific value of a random variable *x*. Knowing there have been *N* experiments and that the value *i* occurs *n<sub>i</sub>* times, then:

![Laplace Smoothing](assignment1/Laplace_smoothing.png)

Where *v* is the number of values of the attribute *x* and *a* is a parameter that express how the data needs to be trusted from the program.
With a value of a greater than zero, it’s possible to avoid the problem of zero probability.
In order to implement this algorithm, a few changes to the code are necessary:
- First of all, the program needs to know the number of levels of each at￾tribute. This information shall be given as an additional row of the training set;
- The program must remove the row with the number of levels before it starts to work on the training set.
  Also, if the program doesn’t know at prior the number of levels for each attribute, it shall count the number of unique level from the training set;
- Ultimately, if the Laplace smoothing function is active, the updated way of computing the probability of observing a specific value needs to be used.

## 2) Linear Regression
For the II Assignment, there is a report into the folder that explains what it consists of

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
