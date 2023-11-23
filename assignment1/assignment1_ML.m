%% Task 1: Pre processing
% Select and load the dataset
filename = 'weather_data.txt';
dataset = importdata(filename);

% Initially occurs the replace one or more consecutive spaces with a single
% space in the dataset; it ensures that there is only one space between
% words in the dataset
dataset = regexprep(dataset,' +',' ');

% After that all strings are replaced with integers
dataset = strrep(dataset,'overcast','1');
dataset = strrep(dataset,'rainy','2');
dataset = strrep(dataset,'sunny','3');
dataset = strrep(dataset,'hot','1');
dataset = strrep(dataset,'cool','2');
dataset = strrep(dataset,'mild','3');
dataset = strrep(dataset,'high','1');
dataset = strrep(dataset,'normal','2');
dataset = strrep(dataset,'FALSE','1');
dataset = strrep(dataset,'TRUE','2');
dataset = strrep(dataset,'no','1');
dataset = strrep(dataset,'yes','2');

% Now the first row of the dataset is removed
dataset = dataset(2:size(dataset));

% Dataset conversion from array cell to matrix
dataset = cell2mat(dataset);

% Conversion of the contents of dataset into a numeric array
dataset = str2num(dataset);

%%
% Generation of a random permutation of the integers from 1 to 14: in the
% specific is used to shuffle the numbers from 1 to 14 randomly; so the
% 'rdm_values' variable will contain a random order of these numbers.
rdm_values = randperm(14);

% Select randomly 10 obs for the training set and 4 obs for the test set
trainingSet = dataset(rdm_values(1:10), :);           
testSet = dataset(rdm_values(11:end), :);

%% Task 2 and 3: Naive Bayes Classifier with and without target column

[classification_1, target_1, error_rate_1] = Naive_Bayes_Classifier(trainingSet, testSet);
error_rate_1

testSet_without_target = testSet(:, 1:end-1);
classification_without_target = Naive_Bayes_Classifier(trainingSet, testSet_without_target);

%% Laplace Additive Smoothing with and without target column

% Set the first row as the number of different levels for each variable
trainingSet_2 = [3, 3, 2, 2, 0; trainingSet];          

% Call the classifier
[~, ~, error_rate_2] = Naive_Bayes_Classifier(trainingSet_2, testSet, 'LaplaceSmoothing');
error_rate_2

testSet_without_target_2 = testSet(:, 1:end-1);
classification_without_target_2 = Naive_Bayes_Classifier(trainingSet_2, testSet_without_target_2, 'LaplaceSmoothing');

%% TestSet with attribute values that were not in the trainingSet 
% With the use of the 'try - catch' statement, there are no errors

% It is only necessary to modify the testSet
testSet_unknown_value = testSet;
testSet_unknown_value(2,3) = 8;

% Call the classifier
[~, ~, error_rate_3] = Naive_Bayes_Classifier(trainingSet, testSet_unknown_value);
error_rate_3

% Furthermore, using Laplace there still will be no errors because of
% the algorithm itself

%% Function Naive Bayes Classifier

function [classification, target, error_rate] = Naive_Bayes_Classifier(trainingSet, testSet, type)
% The Naive Bayes Classifier is a probabilistic classifier based on
% applying Bayes theorem with naive independence assumptions between
% the features.
%
% The trainingSet and the testSet must be two matrices with
% the observations as rows and the features as columns.
%
% The number of columns of the trainingSet must be up to 1 column greater
% than the testSet one: this is because the testSet may have or not
% the target column.
%   
% The third argument is used to set the Laplace (Additive) Smoothing
% algorithm.
%
% Legend:
% - classification = Naive_Bayes_Classifier(trainSet, testSet)
%
%   Classifies the obs in the testSet using the trainingSet obs as
%   reference.
%
% - [classification, target, error_rate] = Naive_Bayes_Classifier(trainSet, testSet)
%
%   Classifies the obs in the testSet using the trainingSet obs as
%   reference. Then compute the error rate thanks to the target column
%   given with the testSet. It also returns the target column for
%   clarity.
%
% - classification = Naive_Bayes_Classifier(trainSet, testSet, 'LaplaceSmoothing')
%
%   Classifies the obs in the test set using the trainingSet obs as
%   reference. It uses the Laplace Smoothing algorithm.
%
% - [classification, target, error_rate] = Naive_Bayes_Classifier(trainSet,testSet, 'LaplaceSmoothing')
%
%   Classifies the obs in the testSet using the trainingSet obs as
%   reference. It uses the Laplace Smoothing algorithm. Then compute the 
%   error rate thanks to the target column given with the testSet. 
%   It also returns the target column for clarity.

bool_laplace = 0;
if nargin > 2   % 'nargin' is a function that returns the number of input
                % arguments passed to the current function
    
    if type == 'LaplaceSmoothing'
        bool_laplace = 1;
    end

end

if bool_laplace == 1
    % Save the first row as the number of levels
    v = trainingSet(1,:);
    trainingSet = trainingSet(2:end, :);        
    a = 1;    
end

% Save the size of both Sets
[n, d] = size(trainingSet);
d = d - 1;
[m, c] = size(testSet);

% Checks on the sets
if c < d || nnz(trainingSet < 1) > 0 || nnz(testSet < 1) > 0
    error('Error') 
end

% Compute the number of classes to preallocate 
% the likelihood and the probClasses matrices (for better performance)
classes = length(unique(trainingSet(:, end))); 
likelihood = cell(classes, d);

probClasses = zeros(1, classes);

for class = 1:classes
    % Set the number of occurences as the number of rows of the trainSet
    % matrix with the last value equal to the analyzed class
    N_occurences = size(trainingSet(trainingSet(:, end) == class, :), 1);

    probClasses(class) = N_occurences / n;

    for variable = 1:d
        if bool_laplace
            possible_levels = v(variable);
        else
            possible_levels = length(unique(trainingSet(:, variable)));
        end
       
        % Preallocating for better performance 
        freq = zeros(1, possible_levels);

        for level = 1:possible_levels
            % For each level of each variable, its frequency is equal to:
            % the number of instances of the analyzed class that has
            % the analyzed variable equal to the analyzed level, 
            % divided by the number of instances of the analyzed class.
            % NB: In Laplace Smoothing the 'a' parameter represent how
            % much you trust your prior belief over the data
            if bool_laplace
                freq(level) = (nnz(trainingSet(trainingSet(:, end) == class, variable) == level) + a) / (N_occurences + a * v(variable));
            else
                freq(level) = (nnz(trainingSet(trainingSet(:, end) == class, variable) == level)) / (N_occurences);
            end
            % We dont have to worry with Laplace smoothing but, if the
            % level doesnt exist in the trainingSet, we should set its
            % frequency to 0 or else we will have an error later
        end
        likelihood{class, variable} = freq;
    end
end

for x = 1:m     % For each obs in the testSet 
    X = testSet(x, 1:d);
    for t = 1:classes
        g(x,t) = log(probClasses(t));  % Preallocate for better performance
        for i = 1:d     % For each variable
            freq = log(cell2mat(likelihood(t,i)));
            try 
               g(x,t) = g(x,t) + freq(X(i));
            catch % If the probability does not exist, set to 0
               g(x,t) = 0;
            end
        end
    end
end

classification = ones(x, 1);
for x = 1:m
    for t = 1:classes
        if g(x, t) > g(x, classification(x))
            classification(x) = t;
        end
    end
end

if c == d + 1     % In the testSet there is the target column  
    target = testSet(:, end);
    error_rate = nnz(classification ~= target) / m;        
else
    disp('Target not present in the testSet')
end

end