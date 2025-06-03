function [beta_draws,sigma_draws] = MC_TVPSVQR(y,X,nsave,nburn,quant,prior)
%% mc_tvpsvqr: Function that does flexible estimation of Bayesian quantile regression (used in Monte Carlo study)
%  INPUTS
%    y      LHS variable of interest
%    X      RHS matrix
%  nsave    Number of Gibbs draws to store after convergence
%  nburn    Number of initial Gibbs draws to discard
%  quant    Vector of quantiles of y to estimate
%  prior    Shrinkage prior to use. Choices are:
%                                   1: Normal-iGamma prior (Student t)
%                                   2: SSVS with Normal(0,tau_0) and Normal-iGamma components
%                                   3: Horseshoe prior
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% First version: 04 November 2020
% This version: 07 November 2020
%==========================================================================

[T,p] = size(X);

% create matrices for TVP estimation
Tp = T*p;   
H = speye(Tp,Tp) - sparse(p+1:Tp,1:(T-1)*p,ones(1,(T-1)*p),Tp,Tp);
Hinv = speye(Tp,Tp)/H;
bigG = full(SURform(X)*Hinv);
x = [ones(T,1) bigG];

% ==============| Define quantiles
n_q = length(quant);
tau_sq = 2./(quant.*(1-quant));   
theta = (1-2*quant)./(quant.*(1-quant));
theta_sq = theta.^2;

% ==============| Define priors
% prior for beta
Q    = .01*ones(Tp+1,n_q);
miu  = zeros(Tp+1,n_q);  
if prior == 1
    % Student_T shrinkage prior
    b0 = 0.01;
elseif prior == 2       
    % SSVS prior
    c0 = (0.01)^2;
    tau = 4*ones(Tp+1,n_q);
    pi0 = 0.25;
    b0 = 0.01;
elseif prior == 3   
    % Horseshoe prior
    lambda = 0.1*ones(Tp+1,n_q);     % "local" shrinkage parameters, one for each Tp+1 parameter element of betaD
    tau = 0.1*ones(1,n_q);           % "global" shrinkage parameter for the whole vector betaD
end

% ==============| Initialize vectors
beta = rand(Tp+1,n_q);
betaD = rand(Tp+1,n_q);
beta_mat = zeros(T,p+1,n_q);
sigma_t = 0.1*ones(T,n_q);
h = ones(T,n_q); 
sig = 0.1*ones(n_q,1);

% Storage space for Gibbs draws
beta_draws = zeros(T,p+1,n_q,nsave);
sigma_draws = zeros(T,n_q,nsave);

%% =========| GIBBS sampler starts here
iter = 500;             % Print every "iter" iteration
fprintf('Iteration 0000')
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end
    
    % Sample z
    k1 = sqrt(theta_sq + 2*tau_sq)./abs(y - x*betaD);
    k2 = (theta_sq + 2*tau_sq)./(sigma_t.*tau_sq);
    z  = min(1./random('InverseGaussian',k1,k2,T,n_q),1e+6);

    % Sample stochastic volatility sigma_t
    a1 = 0.1 + 3*T/2;    sse = (y - x*betaD - theta.*z).^2;
    a2 = 0.1 + sum(sse./(2*z.*tau_sq)) + sum(z);       
    sigma2 = 1./gamrnd(a1,1./a2);                            % Sample from inverse-Gamma
    sigma_t = repmat(sigma2,T,1);

    % Sample regression coefficients beta
    u = sqrt(1./(sigma_t.*tau_sq.*z));
    y_tilde = (y - theta.*z).*u;
    for q = 1:n_q
        xnew = x.*u(:,q);   
        betaD(:,q) = randn_gibbs(y_tilde(:,q),xnew,miu(:,q),Q(:,q),T);
        beta(:,q) = [betaD(1,q); Hinv*betaD(2:end,q)];
        beta_mat(:,:,q) = [repmat(beta(1,q),T,1), reshape(beta(2:end,q),p,T)'];
   
        %--draw prior variance
        if prior == 1
            [Q(:,q),~,miu(:,q)] = student_T_prior(betaD(:,q),b0);
        elseif prior == 2
            [Q(:,q),~,miu(:,q),tau(:,q)] = ssvs_prior(betaD(:,q),c0,tau(:,q),b0,pi0);
        elseif prior == 3
            [Q(:,q),~,miu(:,q),lambda(:,q),tau(:,q)] = horseshoe_prior(betaD(:,q),Tp+1,lambda(:,q),tau(:,q));
        end        
    end
    
    if irep > nburn
       beta_draws(:,:,:,irep-nburn) = beta_mat;
       sigma_draws(:,:,irep-nburn) = sigma_t;
    end    
end
fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
end

