function [w1, w0] = Linear_Regres_OneDim(dataset_turkish, type)
% One dimension linear regression
%   Approximate a functional dependency based on measured data.
%   The data matrix must have two columns, one for the Observations and one
%   for the Targets.
%   The goal of the algorithm is to select the parameter that minimize the
%   objective function, employing the least squares method.
%   If a second argument is provided, it is used to determine whether or
%   not the intercept should be included.

    w1num = 0;
    w1den = 0;
    N = length(dataset_turkish);    

    withOffset = 0;
    if nargin > 1
        if type == 'withOffset'
            withOffset = 1;   
        end
    end    
      
    if withOffset        
        x_hat = sum(dataset_turkish(:,1)) / N;
        t_hat = sum(dataset_turkish(:,2)) / N;
    else
        x_hat = 0;
        t_hat = 0;
    end
    
    for i=1:N
        w1num = w1num + (dataset_turkish(i,1) - x_hat) * ...
        (dataset_turkish(i,2) - t_hat);
        w1den = w1den + (dataset_turkish(i,1) - x_hat)^2;
    end

    w1 = w1num / w1den;

    if withOffset
        w0 = t_hat - w1 * x_hat;
    else
        w0 = nan;
    end
end