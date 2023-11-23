%% This MATLAB code must be into the " mnist " folder in order to work!

%% Important:
%   As outlined in the function explanation, the developed kNN classifier
%   will notify the user in case the chosen k value is evenly divisible by
%   the number of classes.
%   This is highly likely to occur as we simplify the initial 10-class
%   problem into a 2-class problem, making every even k value susceptible
%   to triggering the warning.

clear, clc

%% Settings

% Percentage of test set observations to take randomly
percent_test_obj = 0.5; 

% Class to compute the accuracy on
classes = 1:10;

% Vector of k given for this assignment
k = [1,2,3,4,5,10,15,20,30,40,50]; 

%% Data pre-processing

% Loading the whole training set and test set
[X, T] = loadMNIST(0);   
[test_set, test_labels] = loadMNIST(1);

% Taking randomly the specified percentage of the test set
random_indexes = randi(length(test_set), 1, length(test_set) * ...
    percent_test_obj);
test_set = test_set(random_indexes, :);
test_labels = test_labels(random_indexes, :);
    
%% Computation

for i = classes
    % Assign 1 as label of the class to analyze and 0 as label of the
    % others
    T_i = (T == i);
    test_labels_i = (test_labels == i);
    
    fprintf('\nComputing accuracy on the %d° class \n', i);

    % Computing the accuracy in recognizing the i-th class for each one of
    % the k values
    [z, accur] = kNN_classifier(X, T_i, test_set, k, test_labels_i);    
    accuracy(i, :) = accur';   

end

%% Plotting

figure(1)
hold on
for i = classes
        if i == 10
            message = ['Class ', num2str(0)];
        else
            message = ['Class ', num2str(i)];
        end

        plot(k, accuracy(i,:), '-','DisplayName', message)

end
hold off
legend show
title('Classes Accuracy Variation')
xlabel('k')
ylabel('Accuracy')

function [test_classes_hat, accuracy, k] = kNN_classifier(training_set, training_classes, test_set, k, test_classes)
% The k-Nearest Neighbors classifier is a non-parametric classification
% technique. An object is classified by a plurality vote of its neighbors,
% with the object being categorized into the class that is most prevalent
% among its k nearest neighbors.
%
% The value of k can take the form of an integer or a vector of integers.
% When it's a vector, the output provides an estimated class for each
% distinct k within the vector.
% If k = 1, the object is straightforwardly assigned to the class of its
% single nearest neighbor.
% It's important to remember that k must not be less than or equal to zero
% or exceed the total number of objects within the training dataset.
% Additionally, k should not be evenly divisible by the number of classes;
% if this condition arises, a warning will be generated.
%
% test_classes_hat = kNN_classifier(training_set, training_classes, test_set, k)
%      This classifies the objects in the test_set.
%
% [test_classes_hat, accuracy] = kNN_classifier(training_set, training_classes, test_set, k, test_classes)
%      This classifies the objects in the test_set and compute the accuracy
%      using the expected output (test_classes).
% 
% EXAMPLES: 
%   test_classes_hat = kNN_classifier(training_set, training_classes, test_set, [1 2 3])
%       Classifies the test_set three times, with a k that assumes the 
%       value 1, 2 and 3. The output will be a matrix with as many rows as 
%       the number of objects in the test_set and 3 columns. 
%       test_classes_hat(i, j) will be the class of the i-th object 
%       classified keeping in account the j-th value in the k vector.
%
%   [test_classes_hat, accuracy] = kNN_classifier(training_set, training_classes, test_set, 5, test_classes)
%       Classifies the test_set keeping in account the 5-nearest neighbors
%       inside the training_set. It than computes the accuracy using 
%       test_classes.       

    [n, d1] = size(training_set);
    [m, d2] = size(test_set);

    test_classes_hat = nan;
    accuracy = nan;

    % Check if values in k are divisible by the number of classes
    num_classes = nnz(unique(training_classes));
    if nnz(size(k) ~= size(k(mod(k, num_classes) ~= 0)))
        fprintf(['\nWarning, some of the values in k are divisible by ' ...
          'the number of classes. \nThis could cause ties.\n']);
    end

    % Check the number of arguments, the size of d, the value(s) of k
    if nargin < 4
        fprintf('Error, arguments are not sufficient\n');
        return
    end

    if d1 ~= d2 
        fprintf('Error, training set and test set columns do not match\n');
        return
    end

    if nnz(k <= 0) || n < max(k)
        fprintf(['Error, k must be greater than zero and\n ' ...
                'cannot be greater than the number of elements\n ' ...
                'inside the training set\n']);
        return
    end

    % test_classes_hat will have as much column as the k given in input
    test_classes_hat = zeros(m, size(k,2));
    
    % Computing the euclidean distance of every observation in the test set
    % from each observation in the training set.
    % The second output saves the indexes of the max(k) smallest distances.
    [~, I] = pdist2(training_set, test_set, 'euclidean', 'Smallest', max(k));

    % For each observation in the test set
    for i = 1:m
        for kv = 1:length(k)  % For each k value
            % Take the mode of the labels of the first k smallest distances
            % observations in the training set.
            test_classes_hat(i, kv) = mode(training_classes(I(1:k(kv), i)));
        end
    end

    % Check whether the given input arguments are 5
    if nargin == 5

        % If so, use the true test labels to compute the accuracy. 
        % The accuracy change with k, so it needs to be computed for
        % each value of k.
        for kv = 1:length(k)

            % The accuracy is the number of correct estimations on the
            % total number of observation in the test set, multiplied
            % by 100
            accuracy(kv,1) = (nnz(test_classes_hat(:,kv) == test_classes) ...
                / m) * 100 + 0.00;   
        end
    end
end