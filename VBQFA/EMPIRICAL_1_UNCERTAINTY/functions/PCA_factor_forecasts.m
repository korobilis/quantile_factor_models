function [y_hat,F] = PCA_factor_forecasts(y, X, y_lag, h, nf, max_lag)
    
    %% |==== Compute PCA
    [F,~] = extract(zscore(X),nf);
    
    %% |==== Compute coefficients + BIC 
    beta = cell(max_lag,1);
    BIC  = NaN(max_lag,1);
    
    for p = 1:max_lag
        X = [ones(length(y)-h-p,1) F(p+1:end-h,:) y_lag(p+1:end-h,1:p)];
        Y  = y(1+h+p:end,:);
        beta{p} = (X'*X)\X'*Y;
        resid_  = Y-X*beta{p};
        ssr     = resid_'*resid_;
        BIC(p)  = log(ssr/length(Y)) + (p+1)*log(length(Y))/length(Y);
    end
    
    %% |==== Evaluate BIC 
    [~, optimal_lag] = min(BIC);  
    
    %% |==== Compute Forecasts 
    X     = [1, F(end,:), y_lag(end,1:optimal_lag)];
    y_hat = X*beta{optimal_lag};
end

