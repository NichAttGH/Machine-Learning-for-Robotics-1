# Machine Learning 1
The assignments are:
1) Naive Bayes classifier
2) Linear regression
3) kNN Classifier

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

Practically, the returned likelihood matrix is a cell matrix with as many
rows as the number of classes and as many columns as the number of attribute.
Each value of the matrix is an array that stores the frequency of each possible
level of the specific attribute in the specific class. With this data, it’s possible
to compute the overall probability that a given observation belong to a class
rather than another.
