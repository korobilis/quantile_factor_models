function [F,L] = VBPPCA(x, r, maxiter)

%% VBPPCA Variational Bayes estimation of probabilistic PCA model, as in Bishop (1999)
% ====================================================================================
% The model is of the form
%
%           X = F L + e,   e ~ N(0,sigma2)
%
% with priors F ~ N(mu,I), L ~ N(0,alpha) with alpha ~ iGamma(a,b), mu ~ N(0,\infty), 
% and sigma2 ~ iGamma(r0,s0)
% ====================================================================================
% INPUT:
%     x     (n x p) variable of measurements
%     r     Number of factors
%  maxiter  Maximum number of iterations for variational Bayes
%  
% OUTPUT:
%    F    (n x r) matrix of r principal components
%    L    (p x r) loadings matrices of size p x r
% =========================================================================
% Written by Dimitris Korobilis and Maximilian Schroeder
% University of Glasgow and Norwegian BI Business School
% This version: November 2022


% Turned shrinkage off! line 69!

[n,p] = size(x);

% lambda_j ~ N(0,D)
Ddiag = 100*ones(p,r); 
Dinv = repmat(0.01*eye(r),1,1,p);
% sigma ~ InvGam(r0/2,s0/2)
r0 = 1.0e-4;
s0 = 1.0e-4;
% machine learning prior
a = 1.0e-4;
b = 1.0e-4; 

% Initialize parameters
mu_f      = extract(zscore(x),r);
mu        = mean(mu_f);
Sigma_f   = zeros(r,r,n);
mu_l      = zeros(r,p);
Sigma_l   = zeros(r,r,p);
E_sig     = 1;

% EM algorithm settings
Threshold = 1.0e-3;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

while kappa < maxiter && abs(F_new - F_old) > Threshold
    
    % Demean factors prior to estimation
    mu_f = mu_f - mu;

    invSig = (1./E_sig)*eye(n); detsigl = 0;
    for i = 1:p
        %% Update lambda
        Sigma_l(:,:,i) = eye(r)/((mu_f'*invSig*mu_f) + squeeze(Dinv(:,:,i)));
        mu_l(:,i) = Sigma_l(:,:,i)*((mu_f'*invSig*x(:,i)));
        detsigl = detsigl + det(Sigma_l(:,:,i));

%         %% Update Ddiag
%         a_alpha = a + 0.5;
%         b_alpha = b + 0.5*(mu_l(:,i).^2 + diag(Sigma_l(:,:,i)));
%         Ddiag(i,:) = b_alpha./a_alpha;
%         Dinv(:,:,i) = diag(1./(Ddiag(i,:) + 1e-10));
    end
        
    %% Update sigma2
    for i = 1:p; M(:,i) = sum((x(:,i) - mu_f*mu_l(:,i)).^2  + diag(mu_f*Sigma_l(:,:,i)*mu_f')); end
    r_sig = r0 + (n*r)/2;
    s_sig = s0 + sum(sum(M))/2;
    E_sig = (s_sig./r_sig);

    %% Update factors
    invSig = (1./E_sig)*eye(p);
    detsigf = 0;
    for i = 1:n
        Sigma_f(:,:,i) = eye(r)/((mu_l*invSig*mu_l') + eye(r));
        mu_f(i,:) = Sigma_f(:,:,i)*(mu_l*invSig*x(i,:)' + mu');
        detsigf = detsigf + det(Sigma_f(:,:,i));
    end    

    %% Update mean of factors
    mu = mean(mu_f);

    % Get ELBO and go to the next iteration
    %%% ====| Compute the ELBO
    ELBO  = n*sum(log(detsigf)) + p*sum(log(detsigl)) - r_sig*log(s_sig); %- sum(a_alpha*log(b_alpha))
    F_old = F_new;
    F_new = ELBO;
    kappa = kappa+1;
end
F = mu_f;
L = mu_l';
end
