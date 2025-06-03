function [QFKS,QLKS] = RQPCA_factor_only(x,trend,r,quant,est_meth)

% RQPCA Function to compute regularized quantile PCA using two-step Korobilis-Schroeder (KS) algorithm
% INPUTS:
%    x      Data typically of dimensions n x p (obs x variables)
%    r      Number of principal components to estimate
%  quant    Vector/scalar of quantile levels
% est_meth  Method for recovery of factors: 1) PCA, 2) PPCA

[T,p] = size(x);
nq = length(quant);       

% ========| Simple PCA estimates
FPCA = extract(zscore(x),r); 
    
% ========| 
% 1) First extract quantile time-varying trends (local level model)
% trend = zeros(T,p,nq);
% for j = 1:p
%     [trend(:,j,:)] = VBQLL(x(:,j),100,quant);                  
% end

% 2) Next obtain (P)PCAs per quantile level
QFKS = zeros(T,r,nq);
QLKS = zeros(p,r,nq);
for q = 1:nq
    if est_meth == 1      % Use PCA
        [QFKS(:,:,q),QLKS(:,:,q)] = extract(trend(:,:,q),r);          
    elseif est_meth == 2  % Use PPCA
        [QFKS(:,:,q),QLKS(:,:,q)] = VBPPCA(trend(:,:,q),r);
    end
    % Normalize factors to be following PCA   
    for ifac = 1:r
        corrPCAQF = corrcoef(QFKS(:,ifac,q),FPCA(:,ifac));
        QFKS(:,ifac,q)  = sign(corrPCAQF(1,2)).*QFKS(:,ifac,q);
        QLKS(:,ifac,q)  = sign(corrPCAQF(1,2)).*QLKS(:,ifac,q);
    end       
end