function [beta_draws,sigma_draws] = MC_TVPSVMR(y,X,nsave,nburn,prior)
%% mc_tvpsvmr: Function that does flexible estimation of Bayesian "mean" regression (used in Monte Carlo study)
%  INPUTS
%    y      LHS variable of interest
%    X      RHS matrix
%  nsave    Number of Gibbs draws to store after convergence
%  nburn    Number of initial Gibbs draws to discard
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

% ==============| Define priors
% prior for beta
Q    = .01*ones(Tp+1,1);
miu  = zeros(Tp+1,1);  
if prior == 1
    % Student_T shrinkage prior
    b0 = 0.01;
elseif prior == 2       
    % SSVS prior
    c0 = (0.01)^2;
    tau = 4*ones(Tp+1,1);
    pi0 = 0.25;
    b0 = 0.01;
elseif prior == 3   
    % Horseshoe prior
    lambda = 0.1*ones(Tp+1,1);     % "local" shrinkage parameters, one for each Tp+1 parameter element of betaD
    tau = 0.1;           % "global" shrinkage parameter for the whole vector betaD
end

% ==============| Initialize vectors
beta = rand(Tp+1,1);
betaD = rand(Tp+1,1);
sigma_t = 0.1*ones(T,1);
h = ones(T,1); 
sig = 0.1;

% Storage space for Gibbs draws
beta_draws = zeros(T,p+1,nsave);
sigma_draws = zeros(T,nsave);

%% =========| GIBBS sampler starts here
iter = 500;             % Print every "iter" iteration
fprintf('Iteration 0000')
for irep = 1:(nsave+nburn)
    % Print every "iter" iterations on the screen
    if mod(irep,iter) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end

    % Sample stochastic volatility sigma_t
%     yhat = (y - x*betaD);
%     ystar  = log(yhat.^2 + 1e-6);
%     [h, ~] = SVRW(ystar,h,sig,4);  % log stochastic volatility    
%     sigma_t  = exp(h);       % variance
%     r1 = 1 + T - 1;   r2 = 0.01 + sum(diff(h).^2)';
%     sig = 1./gamrnd(r1./2,2./r2);   % sample state variance of log(sigma_t.^2)
    a1 = 0.1 + T/2;    sse = (y - x*betaD).^2;
    a2 = 0.1 + sum(sse);       
    sigma2 = 1./gamrnd(a1,1./a2);                            % Sample from inverse-Gamma
    sigma_t = repmat(sigma2,T,1);   

    % Sample regression coefficients beta
    u = sqrt(1./(sigma_t));
    y_tilde = y.*u;
    xnew = x.*u;   
    betaD = randn_gibbs(y_tilde,xnew,miu,Q,T);
    beta = [betaD(1,1) ; Hinv*betaD(2:end,:)];
    beta_mat = [repmat(beta(1,1),T,1), reshape(beta(2:end,1),p,T)'];
      
    %--draw prior variance  
    if prior == 1
        [Q,~,miu] = student_T_prior(betaD,b0);
    elseif prior == 2
        [Q,~,miu,tau] = ssvs_prior(betaD,c0,tau,b0,pi0);
    elseif prior == 3
        [Q,~,miu,lambda,tau] = horseshoe_prior(betaD,Tp+1,lambda,tau);
    end
    
    if irep > nburn
       beta_draws(:,:,irep-nburn) = beta_mat;
       sigma_draws(:,irep-nburn) = sigma_t;
    end    
end
fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
end

