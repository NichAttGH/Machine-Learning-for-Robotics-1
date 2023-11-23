%% Turkish Dataset
clear, clc

%% Task 1: Get data
filename_turkish = 'turkish.csv';
dataset_turkish = readmatrix(filename_turkish);

%% Task 2: Fit a linear regression model
% 1) One-dimensional problem without intercept on the Turkish stock
%    exchange data:
w = Linear_Regres_OneDim(dataset_turkish);

% Plot results
figure(1)
scatter(dataset_turkish(:,1), dataset_turkish(:,2), 'x')
hold on
plot(dataset_turkish(:,1), w * dataset_turkish(:,1))
hold off


% 2) Compare graphically the solution obtained on different random
% subsets (10%) of the whole data set:

N = length(dataset_turkish);

flag_input_correct = false;
while ~flag_input_correct
    check = input("Digit ' 1 ' if you wanto to select observations from " + ...
        "different ends of the dataset, otherwise digit ' 0 '\n");
    if check == 1 % Select observations from different ends of the dataset
        size_subset = round(N * 0.2);
        dataset_2_begin = dataset_turkish(1:size_subset, :);
        dataset_2_end = dataset_turkish(end - size_subset + 1:end, :);

        rand_subset_1 = randperm(size_subset, round(N * 0.1));
        w1 = Linear_Regres_OneDim(dataset_turkish(rand_subset_1,:));

        rand_subset_2 = randperm(size_subset, round(N * 0.1));
        rand_subset_2 = rand_subset_2 + round(N * 0.8);
        w2 = Linear_Regres_OneDim(dataset_turkish(rand_subset_2,:));

        flag_input_correct = true;
    elseif check == 0  % Select observations randomly across the whole dataset      
        rand_subset_1 = randperm(N, round(N * 0.1));
        w1 = Linear_Regres_OneDim(dataset_turkish(rand_subset_1, :));
    
        rand_subset_2 = randperm(N, round(N * 0.1));
        w2 = Linear_Regres_OneDim(dataset_turkish(rand_subset_2, :));

        flag_input_correct = true;
    else
        disp('Invalid input. Please enter 1 or 0!');
    end
end

% Plot results
figure(2)
scatter(dataset_turkish(:,1), dataset_turkish(:,2), 'x')
hold on
plot(dataset_turkish(:,1), w1 * dataset_turkish(:,1), ...
    dataset_turkish(:,1), w2 * dataset_turkish(:,1))
hold off

%% Task 3: Test regression model
% Repeating for 10 times with different dataset splits
figure(3)
scatter(dataset_turkish(:,1), dataset_turkish(:,2), 'x')
hold on

for i=1:10
    % Taking only 5% of the data
    five_percent = round(N * 0.05);

    rand_indexes = randperm(N);
    five_percent_Indexes = rand_indexes(1:five_percent);
    
    ninety_five_percent_indexes = rand_indexes(five_percent + 1:end);
    ninety_five_percent_data = dataset_turkish(ninety_five_percent_indexes, :);
    
    five_percent_data = dataset_turkish(five_percent_Indexes, :);


    % Re-run number 1 (One-dimensional problem without intercept on the
    % Turkish stock exchange data)
    w = Linear_Regres_OneDim(five_percent_data);
    
    plot(five_percent_data(:,1), w * five_percent_data(:,1))

    % Compute the objective (mean square error) on the training data
    J_MSE(1,i) = Mean_Square_Error_OneDim(five_percent_data, 1, 2, w);

    % Compute the objective of the same models on the remaining 95% of the
    % data
    J_MSE(2,i) = Mean_Square_Error_OneDim(ninety_five_percent_data, 1, 2, w);
end

hold off

figure(4)
plot(1:length(J_MSE), J_MSE(1,:))
hold on
plot(1:length(J_MSE), J_MSE(2,:))
legend({'J-MSE-5%','J-MSE-95%'})