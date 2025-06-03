function [beta] = f_VBDVS(y,x,priors)
% VBDVS Function to perform offline estimation of a time-varying parater regression, using the VBDVS algorithm
% =========================================================================================================================
% INPUT:
%          y   Dependent time series variable, with dimension Tx1
%          x   Predictor variables, with dimension Txp
%          q   Never apply DVS prior on the first q predictors (typically intercept and own AR coefficients). If you want
%              to impose shrinkage on all p predictors then simply set q=0
%    maxiter   Maximum number of iterations, in case of no convergence of ELBO
%      prior   Choose prior. Default is 2, (DVS prior used in the paper)
%              Other choices include:
%                      prior = 1;  Simple state-space model with only "random walk prior", no additional shrinkage prior
%                                      p(beta(t)|beta(t-1) ~ N(beta(t-1),Q(t))
%                                      q(t)                = diag(Q(t))
%                                      p(q(t)^{-1})        ~ Gamma(c0,d0)
%                      prior = 2;  As in prior=1, plus the full DVS prior of the form
%                                      p(beta(t))   ~ (1-gamma(t))*N(0,tau1*c) + gamma*N(0,tau1)
%                                      p(tau1^{-1}) ~ Gamma(g0,h0)
%                                      p(gamma(t))  ~ Bernoulli(pi0(t))
%                                      p(pi0(t))    ~ Beta(1,1)
%      delta   Smoothing factor for volatility estimation
% =========================================================================================================================
% OUTPUT:
%       beta   Time varying regression coefficients, Txp matrix of posterior mean estimates
%          V   Variance of beta, array of dimensions pxpxT
%    sigma_t   Time-varying (stochastic) volatility estimates
%   tv_probs   Vector of (time-varying) posterior selection probabilities, in case you select variable selection 
%              This is a Txp matrix, with elements probabilites between 0-1 indicating the probability that the 
%              respective elements of beta are "important" or not 
% =========================================================================================================================           

% unpack priors
q       = priors.q;
maxiter = priors.maxiter;
prior   = priors.prior;
delta   = priors.delta;
c0      = priors.c0;
h0      = priors.h0;
g0      = priors.g0;

% Copyright Dimitris Korobilis, University of Glasgow
% This version: 25 April, 2020

[T, p] = size(x);

% Priors
% 1) beta_0 ~ N(m0,S0)
m0 = zeros(p,1); 
S0 = 4*eye(p);

% 2) q_t ~ Gamma(ct,dt)
ct = ones(T,p);  %c0 = 25;
dt = ones(T,p);  d0 = 1;

if prior == 1  % No shrinkage prior
    tv_probs = zeros(T,p);
elseif prior == 2 % DVS prior
    gt = ones(T,p);  %g0 = 1;
    ht = ones(T,p);  %h0 = 12;
    tau_1 = gt./ht;
    cons = (0.01)^2;
    tau_0 = cons*tau_1;
    pi0 = 0.1*ones(T,1);
    tv_probs = ones(T,p);
end

% 5) sigma_t ~ Gamma(at,bt)
at = ones(T,1);  a0 = 1e-2;
bt = ones(T,1);  b0 = 1e-2;

% Initialize vectors
sigma_t = 0.00001*ones(T,1);

mtt  = zeros(p,T);
mt1t = zeros(p,T);
mtt1 = zeros(p,T);
Stt  = zeros(p,p,T);
Stt1 = zeros(p,p,T);

lambda_t = zeros(T,p);
q_t      = ones(T,p);
Qtilde   = zeros(p,p,T);
Ftilde   = zeros(p,p,T);

Threshold = 1.0e-4;
F_new     = 0;
F_old     = 1;
kappa     = 1;
offset    = .0015;

%format bank;
%fprintf('Now you are running VBDVS')
%fprintf('\n')
%fprintf('Iteration 0000')
while kappa < maxiter && abs((F_new - F_old)) > Threshold
    % Print iterations   
    %if mod(kappa,10) == 0
     %   fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',kappa)
    %end
    
    % ==================| Update \beta_{t} using Kalman filter/smoother
    % Kalman filter
    for t = 1:T
        Qtilde(:,:,t)  = diag(1./(q_t(t,:) + lambda_t(t,:)));            
        Ftilde(:,:,t)  = Qtilde(:,:,t)*diag(q_t(t,:));
        if t==1
            mtt1(:,t)   = Ftilde(:,:,t)*m0;               
            Stt1(:,:,t) = Ftilde(:,:,t)*S0*Ftilde(:,:,t)';
        else
            mtt1(:,t)   = Ftilde(:,:,t)*mtt(:,t-1);
            Stt1(:,:,t) = Ftilde(:,:,t)*Stt(:,:,t-1)*Ftilde(:,:,t)' + Qtilde(:,:,t);
        end
        
        Sx              = Stt1(:,:,t)*x(t,:)';         
        Kt              = Sx/(x(t,:)*Sx + sigma_t(t,1));
        mtt(:,t)        = mtt1(:,t) + Kt*(y(t,:) - x(t,:)*mtt1(:,t));
        Stt(:,:,t)      = (eye(p) - Kt*x(t,:))*Stt1(:,:,t);
    end
    
    % Fixed interval smoother    
    mt1t = zeros(p,T);   mt1t(:,t) = mtt(:,t);
    St1t = zeros(p,p,T); St1t(:,:,t) = Stt(:,:,t);
    for t = T-1:-1:1
        C = (Stt(:,:,t)*Ftilde(:,:,t+1))/Stt1(:,:,t+1);        
        mt1t(:,t)   = mtt(:,t) + C*(mt1t(:,t+1) - mtt1(:,t+1));   
        St1t(:,:,t) = Stt(:,:,t) + C*(St1t(:,:,t+1) - Stt1(:,:,t+1))*C';
    end

    % =====================| Update hyperparameters
    for t = 1:T
        eyeF            = (eye(p) - 2*Ftilde(:,:,t))';
        if t == 1
            D           = St1t(:,:,t) + mt1t(:,t)*mt1t(:,t)' + (S0 + m0*m0')*eyeF;
        else
            D           = St1t(:,:,t) + mt1t(:,t)*mt1t(:,t)' + (St1t(:,:,t-1) + mt1t(:,t-1)*mt1t(:,t-1)')*eyeF;
        end
        
        if prior == 2              
            gt(t,:)     = g0 + 0.5;
            ht(t,:)     = h0 + (mt1t(:,t).^2)./2;
            tau_1(t,:)  = ht(t,:)./gt(t,:);
            tau_0(t,:)  = cons*tau_1(t,:);
            l_0         = lnormpdf(mt1t(:,t),zeros(p,1),sqrt(tau_0(t,:))') + 1e-20;
            l_1         = lnormpdf(mt1t(:,t),zeros(p,1),sqrt(tau_1(t,:))') + 1e-20;
            gamma       = 1./( 1 + ((1-pi0(t))./pi0(t)).* exp(l_0 - l_1) );
            pi0(t)        = (1 + sum(gamma==1))/(2+p);
            tv_probs(t,:) = gamma;            
            lambda_t(t,:) = 1./(((1-gamma).^2).*tau_0(t,:)' + (gamma.^2).*tau_1(t,:)');%((1-gamma).^2).*(1./tau_0(t,:))' + (gamma.^2).*(1./tau_1(t,:))';
            lambda_t(t,1:q) = 0;
        end
        
        % State variances Q_{t}
        ct(t,:)     = c0 + 0.5;
        dt(t,:)     = d0 + max(1e-10,diag(D)/2);
        q_t(t,:)    = ct(t,:)./dt(t,:);
    end       
    
    % =============| Next update stochastic volatilities       
    % Filter volatilities
    s_tinv = zeros(T,1);
    for t = 1:T           
        temp = x(t,:)*( mt1t(:,t)*mt1t(:,t)' + St1t(:,:,t) )*x(t,:)' - 2*x(t,:)*mt1t(:,t)*y(t,:) + (1 + offset)*y(t,:)*y(t,:)';
        if t == 1
            at(t,:) = a0 + 0.5;
            bt(t,:) = b0 + temp/2;
        else
            at(t,:) = delta*at(t-1,:) + 0.5;
            bt(t,:) = delta*bt(t-1,:) + temp/2;
        end
        s_tinv(t,:) = at(t,:)./bt(t,:);
    end
    % Smooth volatilities
    phi = zeros(T,1); phi(T,:) = at(T,:)./bt(T,:);
    for t=T-1:-1:1
        phi(t,:) = (1-delta)*s_tinv(t,:) + delta*phi(t+1,:);
    end
    sigma_t = 1./phi;

    % Get ELBO and go to the next iteration
    F_old = F_new;
    F_new = norm(lnormpdf(y,sum(x.*mt1t',2),sigma_t)) - sum(klgamma(gt,ht,g0,h0)) - sum(klgamma(ct,dt,c0,d0));

    kappa = kappa + 1;
end
%fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)

beta  = mt1t';
V     = St1t;
end


function [kl] = klgamma(pa,pb,qa,qb)

n = max([size(pb,2) size(pa,2)]);

if size(pa,2) == 1, pa = pa*ones(1,n); end
if size(pb,2) == 1, pb = pb*ones(1,n); end
qa = qa*ones(1,n); qb = qb*ones(1,n);

kl = sum( pa.*log(pb)-gammaln(pa) ...
         -qa.*log(qb)+gammaln(qa) ...
	 +(pa-qa).*(psi(pa)-log(pb)) ...
	 -(pb-qb).*pa./pb ,2);
end


function y = lnormpdf(x,mu,sigma)

if nargin<1
    error(message('stats:normpdf:TooFewInputs'));
end
if nargin < 2
    mu = 0;
end
if nargin < 3
    sigma = 1;
end

% Return NaN for out of range parameters.
sigma(sigma <= 0) = NaN;

try
    y = log(1./(sqrt(2*pi) .* sigma)) + (-0.5 * ((x - mu)./sigma).^2);
catch
    error(message('stats:normpdf:InputSizeMismatch'));
end

end

