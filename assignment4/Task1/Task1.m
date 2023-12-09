%% Lab 4: neural networks
%% Task 1: Feedforward multi-layer networks (multi-layer perceptrons)
% Execute the tutorial:
% 
% <https://it.mathworks.com/help/nnet/gs/classify-patterns-with-a-neural-network.html 
% Matlab tutorial: Classify Patterns with a Neural Network>
%
% *Create result tables* or *use the confusion matrices automatically generated 
% by the Matlab apps*, obtaining experimental results for a couple different problems 
% (data sets) and with different design choices (number of layers, number of units 
% per layer).
%% 1) Dataset pre-processing:
% There are two different datasets with two different pre-processes.

clc, clear

disp('Do you want to load iris dataset or wine dataset?')
choice = input('Digit " i " for iris dataset or " w " for wine dataset\n', 's');
if choice == 'i'
    % Remember that the file will be read only if the variable ' filename '
    % will have the whole file path
    filename = 'C:\Users\nicho\OneDrive\Desktop\Università\1° Anno\ML1\Attolino-Lab4\Task1\iris_data.txt';
    dataset = importdata(filename);
    dataset = strrep(dataset, 'Iris-setosa', '1');
    dataset = strrep(dataset, 'Iris-virginica', '2');
    dataset = strrep(dataset, 'Iris-versicolor', '3');
    dataset = cell2mat(dataset);
    dataset = str2num(dataset);
elseif choice == 'w'
    % Remember that the file will be read only if the variable ' filename '
    % will have the whole file path
    filename = 'C:\Users\nicho\OneDrive\Desktop\Università\1° Anno\ML1\Attolino-Lab4\Task1\wine_data.txt';
    dataset = importdata(filename);
    dataset = [dataset(:, 2:end), dataset(:, 1)];
end
%% 
% The Neural Net Pattern Recognition app needs two matrices:
% - The predictors one, which has the input data necessary to the prediction;
% - The responses one, which has the corresponding class for each observation.

x = dataset(:, 1:end-1);

t(:, 1) = dataset(:, end) == 1;
t(:, 2) = dataset(:, end) == 2;
t(:, 3) = dataset(:, end) == 3;
%% 
% The functions expect patterns as columns and variables as rows:

x = x';
t = t';
%% 
% Choose a Training Function:
% - 'trainlm' is usually fastest;
% - 'trainbr' takes longer but may be better for challenging problems;
% - 'trainscg' uses less memory, so is suitable in low memory situations.

train_function = 'trainscg'; 
%% 
% Create a Pattern Recognition Network:

hidden_layer_size = 10;
net = patternnet(hidden_layer_size, train_function);
%% 
% Setup Division of Data for Training, Validation, Testing:

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
%% 
% Train the Network:

[net,tr] = train(net, x, t);
%% 
% Test the Network:

y = net(x);
e = gsubtract(t, y);
performance = perform(net, t, y);
t_indices = vec2ind(t);
y_indices = vec2ind(y);
percentErrors = sum(t_indices ~= y_indices) / numel(t_indices);
%% 
% View the Network:
%
% view(net)
%
% Analyze Results
% The ' Model Summary ' contains information about the training algorithm and 
% the training results for each dataset.
% It's accessible in his raw form by reading the ' tr ' variable. 
% 
% Confusion Matrix

plotconfusion(t, y);
%% 
% It' possibile to see the accuracy of the network outputs by looking the
% numbers of correct classifications in the green (diagonal) squares.
% Those show when the output class matched with the target class.
% By looking to the red squares, it's possible to see as the network
% misplaced some observations assigning another class.
% 
% ROC Curve
% 
% The ROC curve allow additional verification of network performance. 

plotroc(t, y);
%% 
% The colored lines represent the ROC curves. The ROC curve is a plot of the 
% true positive rate (sensitivity) versus the false positive rate
% (1 - specificity) as the threshold is varied. 
% 
% A perfect test would show points in the upper-left corner, with 100%
% sensitivity and 100% specificity.
% For this problem, the network performs very well.