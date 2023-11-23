function J_MSE = Mean_Square_Error_OneDim(dataset, xColumn, tColumn, w1, w0)
% Function to compute the MSE for the one-dimensional problem
% Input:
%       1) dataset
%       2) observation column
%       3) target column
%       4) slope to test
%       5) offset to test

    if nargin < 5
        w0 = 0;
    end

    N = height(dataset);
    T = dataset(:, tColumn);
    Y = w1 * dataset(:, xColumn) + w0;
    J_MSE = (1 / N) * sum((T - Y).^2);

end