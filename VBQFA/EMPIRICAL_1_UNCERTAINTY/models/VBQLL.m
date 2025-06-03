function [beta,sigma] = VBQLL(y,maxiter,quant,prior)

%% VBQLL Variational Bayes quantile local level model estimation
% =========================================================================
%                                  INPUT:
%     y     (T x 1) time series to extract quantile trends
%  maxiter  Maximum number of iterations for variational Bayes
%   quant   Vector of quantiles (a monotonic grid of values, between 0 and 1)
%   prior   1: Uniform prior; 2: Sparse Bayesian learning prior
%  
%                                 OUTPUT:
%   beta    (T x q) matrix of q quantile trends
%  sigma    variance of the measurement error for each quantile level
% =========================================================================
% Written by Dimitris Korobilis and Maximilian Schroeder
% University of Glasgow and Norwegian BI Business School
% This version: November 2022

% Define median quantile to evaluate ELBO
qeval = find((quant.*100)./100 == 0.5);
if isempty(qeval); qeval = quant(1); end

% ==============| Setup local-level model structure
n = length(y);
p = 1; np = n*p;
x = [tril(ones(n))];

% ==============| Define quantiles
nq = length(quant);
k2_sq = 2./(quant.*(1-quant));   
k1 = (1-2*quant)./(quant.*(1-quant));
k1_sq = k1.^2;

% ==============| Define priors
% beta_j ~ N(0,D)
Ddiag = 1000*ones(np,nq); 
Dinv = repmat(0.001*eye(np),1,1,nq);
% sigma ~ InvGam(r0/2,s0/2)
r0 = 1.0e-4;
s0 = 1.0e-4;
% Sparse Bayesian learning prior
a = 1.0e-4;
b = 1.0e-4; 

% Initialize parameters
mu_b      = zeros(np,nq);
Sigma_b   = zeros(np,np,nq);
E_sig     = ones(1,nq);
E_z       = ones(n,nq);
E_iz      = ones(n,nq);

% EM algorithm settings
Threshold = 1.0e-6;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

%% Start Variational Bayes iterations
% format bank;
% fprintf('\n')
while kappa < maxiter && abs(F_new - F_old) > Threshold
%     % Print iterations   
%     if mod(kappa,25) == 1        
%       fprintf('%4f   %4f \n',norm((F_new - F_old)),kappa)        
%     end
    
    %% Update beta          
    u = sqrt(E_iz./(E_sig.*k2_sq));
    y_tilde = (y.*u - k1./(E_sig.*k2_sq));   
    for q = 1:nq
        xnew = x.*u(:,q);
        D = diag(Ddiag(:,q));
        U = bsxfun(@times,Ddiag(:,q),xnew');
        Sigma_b(:,:,q) = D - (U/(eye(n) + xnew*U))*U';
        mu_b(:,q) = Sigma_b(:,:,q)*(xnew'*y_tilde(:,q));
    end   
    
    %% Update prior covariance matrix D
    if prior == 2
        for q = 1:nq           
            Ddiag(:,q) = b + 0.5*(mu_b(:,q).^2 + diag(Sigma_b(:,:,q)))/(a + 0.5);  
            Ddiag(1,q) = 100;
            Dinv(:,:,q) = diag(1./(Ddiag(:,q) + 1e-10));   
        end
    end

    %% Update z
    chi_z = (k1_sq./k2_sq + 2)./E_sig;
    M = zeros(n,nq);
    for q = 1:nq
        M(:,q) = (y - x*mu_b(:,q)).^2 + diag(x*Sigma_b(:,:,q)*x');
    end
    psi_z = (M./(k2_sq.*E_sig));
    E_z   = (sqrt(psi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(chi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1));
    E_iz  = (sqrt(chi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(psi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1)) - 1./psi_z;
    
    %% Update sigma2
    r_sig = r0 + 3*n;
    s_sig = s0 + sum((E_iz.*M)./(2*k2_sq) - (k1.*(y - x*mu_b)./k2_sq) + (1 + k1_sq./(2*k2_sq)).*E_z);
    E_sig = (s_sig./r_sig);  
    
    % Get ELBO and go to the next iteration
    %%% ====| Compute the ELBO
    q = qeval;
    A = (r_sig+2)/2*(psi(r_sig/2)-log(s_sig(:,q)/2))-(E_sig(:,q)/(2*k2_sq(:,q)))*(sum(k1_sq(:,q).*E_z(:,q)-2*k1(:,q).*(y - x*mu_b(:,q)) + E_iz(:,q).*(M(:,q))));   
    B = -E_sig(:,q)*sum(E_z(:,q)) - s_sig(:,q)/2*E_sig(:,q) - sum((mu_b(:,q)+diag(Sigma_b(:,:,q)))./Ddiag(:,q) + 0.5*log(Ddiag(:,q)));   
    L = A+B;
    Ha = 0.5*log(det(Sigma_b(:,:,q))) + 0.5*r_sig + log(0.5*s_sig(:,q)) + gammaln(0.5*r_sig) -(1 + 0.5*r_sig)*psi(0.5*r_sig); % issue with r_sig causing inf or perhaps gamma function does not make sense?   
    Hb = sum(-0.25*(log(chi_z(:,q))-log(psi_z(:,q))) + log(2*besselk(1/2,sqrt(psi_z(:,q).*chi_z(:,q)),1)) +0.5*(psi_z(:,q).*E_iz(:,q)+chi_z(:,q).*E_z(:,q)));
    H = Ha+Hb;
    ELBO = L + H; 
    F_old = F_new;
    F_new = ELBO;
    kappa = kappa + 1;    
end
beta  = cumsum(mu_b);
sigma = E_sig;
end
