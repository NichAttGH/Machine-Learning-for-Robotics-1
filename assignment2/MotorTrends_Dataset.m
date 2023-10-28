%% Motors Trends car dataset
clear, clc

%% Task 1: Get data
filename_motor_trends = 'mtcarsdata.csv';
dataset_motor_trends = readmatrix(filename_motor_trends);

% It's useful to split the dataset in multiple column vectors:
[mpg, disp, hp, weight] = Motor_Trends_Car_load(dataset_motor_trends);

%% Task 2: Fit a linear regression model
% 1) One-dimensional problem with intercept on the Motor Trends car data,
% using columns mpg and weight:

figure(1)
scatter(weight, mpg)

[w1, w0] = Linear_Regres_OneDim([weight, mpg], 'withOffset');

hold on
plot(weight, w1 * weight + w0)
hold off

% 2) Multi-dimensional problem on the complete Motor Trends car data,
% using all four columns (predict mpg with the other three columns):

X = [disp, hp, weight];
T = mpg;

w = (inv(X' * X) * X') * T; % Moore-Penrose pseudoinverse of X * T

%% Task 3: Test regression model
% Repeating for 10 times with different dataset splits
figure(2)
scatter(weight, mpg)  % Whole dataset
hold on

for i=1:10
    % Taking only 5% of the data:
    N = length(dataset_motor_trends);
    five_percent = round(N * 0.05);
    rand_indexes = randperm(N);
    five_percent_indexes = rand_indexes(1 : five_percent);
    ninety_percent_indexes = rand_indexes(five_percent + 1 : end);
    
    five_percent_data = dataset_motor_trends(five_percent_indexes, :);
    ninety_percent_data = dataset_motor_trends(ninety_percent_indexes, :);
    
    [mpg, ~, ~, weight] = Motor_Trends_Car_load(five_percent_data);


    % Re-run number 1 (One-dimensional problem with intercept on the Motor
    % Trends car data, using columns mpg and weight)
    [w1, w0] = Linear_Regres_OneDim([weight,mpg], 'withOffset');
    
    plot(weight, w1 * weight + w0)

    % Compute the objective (mean square error) on the training data
    J_MSE(1,i) = Mean_Square_Error_OneDim(five_percent_data, 5, 2, w1, w0);

    % Compute the objective of the same models on the remaining 95% of the
    % data
    J_MSE(2,i) = Mean_Square_Error_OneDim(ninety_percent_data, 5, 2, w1, w0);


    % 2) Multi-dimensional problem on the complete Motor Trends car data,
    % using all four columns (predict mpg with the other three columns)
    [mpg, disp, hp, weight] = Motor_Trends_Car_load(five_percent_data);
    X_five_percent = [disp, hp, weight];
    T_five_percent = mpg;

    % Moore-Penrose pseudoinverse of X multiplied by T
    w = (inv(X_five_percent' * X_five_percent) * X_five_percent') * T_five_percent;

    Y_five_percent = X_five_percent * w; 

    [mpg, disp, hp, weight] = Motor_Trends_Car_load(ninety_percent_data);
    X_ninety_percent = [disp, hp, weight];
    T_ninety_percent = mpg;

    Y_ninety_percent = X_ninety_percent * w;

    % Compute the objective (mean square error) on the training data
    J_MSE(3,i) = 1/2 * norm(Y_five_percent - T_five_percent)^2;

    % Compute the objective of the same models on the remaining 95% of the
    % data
    J_MSE(4,i) = 1/2 * norm(Y_ninety_percent - T_ninety_percent)^2;
end

hold off

figure(3)
plot(1:length(J_MSE), J_MSE(1,:), 1:length(J_MSE), J_MSE(2,:))
legend({'J-MSE-5%','J-MSE-95%'})
title('One-dimensional problem MSE')

figure(4)
plot(1:length(J_MSE), J_MSE(3,:), 1:length(J_MSE), J_MSE(4,:))
legend({'J-MSE-5%','J-MSE-95%'})
title('Multi-dimensional problem MSE')

% To display the average of the 4 J_MSE computed
% mean(J_MSE(1,:))
% mean(J_MSE(2,:))
% mean(J_MSE(3,:))
% mean(J_MSE(4,:))

function  [mpg, disp, hp, weight] = Motor_Trends_Car_load(dataset_motor_trends)
    mpg = dataset_motor_trends(:,2);
    disp = dataset_motor_trends(:,3);
    hp = dataset_motor_trends(:,4);
    weight = dataset_motor_trends(:,5);
end