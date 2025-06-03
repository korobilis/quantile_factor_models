function [F,L] = VBFA(x, r, maxiter)

%% VBFA Variational Bayes Factor Analysis model estimation
% =========================================================================
% INPUT:
%     x     (n x p) variable of measurements
%     r     Number of factors
%  maxiter  Maximum number of iterations for variational Bayes
%  
% OUTPUT:
%     F    (n x r) matrix of r factors
%     L    (p x r) loadings matrices of size p x r
% =========================================================================
% Written by Dimitris Korobilis and Maximilian Schroeder
% University of Glasgow and Norwegian BI Business School
% This version: November 2022

[n,p] = size(x);

% ==============| Define priors
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
lev_f     = mean(mu_f);
Sigma_f   = zeros(r,r,n);
mu_l      = zeros(r,p);
Sigma_l   = zeros(r,r,p);
E_sig     = ones(p,1);

% EM algorithm settings
Threshold = 1.0e-6;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

while kappa < maxiter && max(abs((F_new - F_old))) > Threshold
    
    % Demean factors prior to estimation
    mu_f = mu_f - mu;
    
    detsigl = 0;
    for i = 1:p
        %% Update lambda
        Sigma_l(:,:,i) = eye(r)/((mu_f'*mu_f)./E_sig(i,:) + squeeze(Dinv(:,:,i)));
        mu_l(:,i) = Sigma_l(:,:,i)*((mu_f'*x(:,i))./E_sig(i,:));
        detsigl = detsigl + det(Sigma_l(:,:,i));
        
        %% Update Ddiag
        a_alpha = a + 0.5;
        b_alpha = b + 0.5*(mu_l(:,i).^2 + diag(Sigma_l(:,:,i)));
        Ddiag(i,:) = b_alpha./a_alpha;
        Dinv(:,:,i) = diag(1./(Ddiag(i,:) + 1e-10));

        %% Update sigma2
        r_sig = r0 + n/2;
        s_sig = s0 + sum((x(:,i) - mu_f*mu_l(:,i)).^2  + diag(mu_f*Sigma_l(:,:,i)*mu_f'));
        E_sig(i,:) = (s_sig./r_sig);
    end
        
    %% Update factors
    invSig = diag(1./E_sig); detsigf = 0;
    for i = 1:n
        Sigma_f(:,:,i) = eye(r)/((mu_l*invSig*mu_l') + eye(r));
        mu_f(i,:) = Sigma_f(:,:,i)*(mu_l*invSig*x(i,:)' + mu');
        detsigf = detsigf + det(Sigma_f(:,:,i));        
    end    

    %% Update mean of factors
    mu = mean(mu_f);

    % Get ELBO and go to the next iteration
    %%% ====| Compute the ELBO      
    ELBO  = n*sum(log(detsigf)) + p*sum(log(detsigl)) - sum(a_alpha*log(b_alpha)) - r_sig*log(s_sig); 
    F_old = F_new;
    F_new = ELBO;    
    kappa = kappa+1;
end
F = mu_f;
L = mu_l';
end
