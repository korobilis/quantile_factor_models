function [QFKS] = RQPCA(x,r,quant,prior)

% RQPCA Function to compute regularized quantile PCA using two-step Korobilis-Schroeder (KS) algorithm
% ======================================================================================================
% INPUTS:
%    x      Data typically of dimensions n x p (obs x variables)
%    r      Number of principal components to estimate
%  quant    Vector/scalar of quantile levels
%  prior    Prior used for quantile trend extraction, 1: Uniform ; 2: Sparse Bayesian Learning
%
% OUTPUTS:
%  QFKS    Quantile factors based on PCA in second-step
% ======================================================================================================
% Written by Dimitris Korobilis and Maximilian Schroeder
% University of Glasgow and Norwegian BI Business School
% This version: November 2022

[T,p] = size(x);
nq = length(quant);       

% ========| Simple PCA estimates
FPCA = extract(zscore(x),r); 
    
% ========| 
% 1) First extract quantile time-varying trends (local level model)
trend = zeros(T,p,nq);
for j = 1:p
    [trend(:,j,:)] = VBQLL(x(:,j),300,quant,prior);
end
warning off
% 2) Next obtain (P)PCAs per quantile level
QFKS = zeros(T,r,nq);
QLKS = zeros(p,r,nq);
for q = 1:nq
    % Use PCA       
    [QFKS(:,:,q),QLKS(:,:,q)] = extract(trend(:,:,q),r); 

    % Normalize factors to follow PCA estimates
    for ifac = 1:r
        corrPCAQF = corrcoef(QFKS(:,ifac,q),FPCA(:,ifac));
        QFKS(:,ifac,q)  = sign(corrPCAQF(1,2)).*QFKS(:,ifac,q);
        QLKS(:,ifac,q)  = sign(corrPCAQF(1,2)).*QLKS(:,ifac,q);
  
    end
end
warning on