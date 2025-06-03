function [trend] = extract_quantile_trends(X,X_fore,quants,nq,prior,gamma)

    [~, p] = size(X);

    XX = [X; X_fore];
    T  = size(XX,1);
    
    trend = zeros(T,p,nq);
    for ivar = 1:p                                                     %maxiter,quant,tvp_reg,prior,gamma_
        [trend(:,ivar,:),~,~,~] = VBTVPQR(zscore(XX(:,ivar)),ones(T,1),300,quants,1,prior,gamma);
    end

end

