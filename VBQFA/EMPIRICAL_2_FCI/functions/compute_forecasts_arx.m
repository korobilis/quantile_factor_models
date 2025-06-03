function [y_hat] = compute_forecasts_arx(y, F, y_lag, h, ar_lag)
    
    %% |==== Compute coefficients in sample
    X = [ones(length(y)-h-ar_lag,1) F(ar_lag+1:end-h,:) y_lag(ar_lag+1:end-h,1:ar_lag)];
    Y  = y(1+h+ar_lag:end,:);
    beta = (X'*X)\X'*Y;

    %% |==== Compute Forecasts 
    if isempty(F)
        X = [1, y_lag(end,:)];
    else
        X     = [1, F(end,:), y_lag(end,:)]; 
    end
    y_hat = X*beta;
end