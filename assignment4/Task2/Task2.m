%% Lab 4: neural networks

% N.B.: The files ' Task2.m ' and ' plotcl.m ' must be in ' mnist ' folder
% in order to work!!!

%% Task 2: Autoencoder
% The simplest *autoencoder network* is a multi-layer perceptron neural
% network which has one input layer, one hidden layer
% (n.hidden units < n.inputs), and one output layer
% (n.output units = n.inputs).
% 
% It is trained using the same pattern as both the input and the target.
% So for instance input pattern x(l, :) has target t(l, :) = x(l, :)
% (the same as the input).
% 
% Note that in this case we don't have any classes or other mapping
% to learn. 
% This is a special case of unsupervised training.
% In fact, it is sometimes called " self-supervised ", since the target we
% use is the input pattern itself.
% 
% Train a multilayer perceptron as an autoencoder for the MNIST data.
% Training an autoencoder only amounts to using a multi-layer perceptron
% neural network for data prepared in a special way (target = input);
% in its basic form it is not a different neural network algorithm.
% However, Matlab provides a separate function that is used as follows:

% myAutoencoder = trainAutoencoder(myData,nh);
% myEncodedData = encode(myAutoencoder,myData);
%
% where nh = size (number of units) in the hidden layer.
% 
% An autoencoder learns an internal, compressed representation for the
% data. 
% The interesting output, therefore, is the value of its hidden layer.
% What we hope is that similar patterns will have similar representations;
% for instance, we hope that images representing a " 1 " will correspond
% to very similar representations, and quite similar to " 7 " but different
% from " 0 " or " 8 ".
% In other words, that the network will learn the classes*. 

clear, clc

% Hidden units
nh = 2;

% Classes pair to test
classes_pair = [1 8; 10, 8; 10, 1; 2, 6; 2, 9; 4, 1; 1, 6]; 

for i = 1:size(classes_pair, 1)  % Repeats for every pair in 'classes'
%% 
% Create a training set with only 2 classes:

    [data, target] = data_pre_processing(classes_pair(i, 1), classes_pair(i, 2), 0.6);
%% 
% Train an autoencoder on the new, reduced training set:

    myAutoencoder = trainAutoencoder(data, nh, 'MaxEpochs', 2000);
%% 
% Encode the different classes using the encoder obtained:

    % Take the data to encode
    [data, target] = data_pre_processing(classes_pair(i, 1), classes_pair(i, 2), 0.05);

    myEncodedData = encode(myAutoencoder, data);
%% 
% Plot the data using the "plotcl" function

    figure(i);

    % Adding the two targets to set the plots legend
    % NB: the plotcl function has been modified
    target = [classes_pair(i, 1), classes_pair(i, 2), target];  
    plotcl(myEncodedData', target');
end

function [data, target] = data_pre_processing(class1, class2, percent)
% Data Pre-processing
%   The function will create a matrix with the desired observations and
%   will order the matrix randomly.
%   Input:
%       - two classes 
%       - percentage of the available observations to use
 
    if nargin < 3
        percent = 1;
    end

    % Take the desired observations
    num_obs = percent * 5000;
    X1 = loadMNIST(0, class1);   
    X2 = loadMNIST(0, class2);
    X1 = X1(1:num_obs, :)';
    X2 = X2(1:num_obs, :)';
    data = [X1, X2];
    
    % Set the target vector
    target = ones(1, num_obs * 2);
    target(1, 1:num_obs) = 1;
    target(1, num_obs + 1:end) = 2;

    % Randomize the order
    random_indexes = randi(size(data, 2), 1, size(data,2));
    data = data(:, random_indexes);
    target = target(1, random_indexes);
end