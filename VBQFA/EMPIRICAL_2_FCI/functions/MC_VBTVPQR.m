function [beta_mat,Sigma_b,sigma,Ddiag,d2,d3,d4] = MC_VBTVPQR(y,X,maxiter,quant,params)
%% tvpsvqr: Function that does flexible estimation of Bayesian quantile regression
%  INPUTS
%    y      LHS variable of interest, measured for t=h+1:T, h is forecast horizon
%    X      RHS matrix (typically intercept, lags of y and lagged exogenous predictors, measured for t=1:T-h)
% maxiter   Maximum number of EM-type iterations
%  quant    Vector of quantiles of y to estimate
% tvp_reg   0: Constant parameter regression; 1: TVP regression; 2: TVP only on intercept
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% First version: 15 June 2022
% This version: 15 June 2022
%==========================================================================

[T,p] = size(X);

% create matrices for TVP estimation
Tp = T*p;   
H = speye(Tp,Tp) - sparse(p+1:Tp,1:(T-1)*p,ones(1,(T-1)*p),Tp,Tp);
Hinv = speye(Tp,Tp)/H;
bigG = full(SURform(X)*Hinv);

if params.addconstant == 1
    x = [ones(T,1) bigG];
else
    x = bigG;  
    p = p-1; %adjust for missing constant 
    Tp = Tp-1;
end

% ==============| Define quantiles
nq = length(quant);
k2_sq = 2./(quant.*(1-quant));   
k1 = (1-2*quant)./(quant.*(1-quant));
k1_sq = k1.^2;

% ==============| Define priors
% beta_j ~ N(0,D)
Ddiag = ones(Tp+1,nq);
Dinv = repmat(eye(Tp+1),1,1,nq);

% D ~ Horseshoe+
Ab = ((100./(Tp+1))).^2;

% sigma ~ InvGam(r0/2,s0/2)
r0 = .01;
s0 = .01;

% Initialize parameters
mu_b      = zeros(Tp+1,nq);
Sigma_b   = zeros(Tp+1,Tp+1,nq);
E_sig     = ones(1,nq);
E_z       = ones(T,nq);
E_iz      = ones(T,nq);
d4        = ones(1,nq);
d3        = ones(Tp+1,nq);   
d2        = ones(Tp+1,nq);
beta      = zeros(Tp+1,nq);
beta_mat  = zeros(T,p+1,nq);

% EM algorithm settings
Threshold = 1.0e-4;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

%% Start Variational Bayes iterations
format bank;
%fprintf('Now you are running VBVS')
%fprintf('\n')
fprintf('Iteration 0000')
while kappa < maxiter && norm((F_new - F_old)) > Threshold
    % Print iterations   
    if mod(kappa,10) == 0
        fprintf('%4f   %4f \n',norm((F_new - F_old)),kappa)
        %fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',kappa)
        
    end
    
    
    %% Update beta
    for q = 1:nq
        Sigma_b(:,:,q) = eye(Tp+1)/((x'*diag(E_iz(:,q))*x)./k2_sq(q) + Dinv(:,:,q));
        mu_b(:,q) = Sigma_b(:,:,q)*((x'*diag(E_iz(:,q))*y)./k2_sq(q) - (k1(q)/k2_sq(q)).*sum(x)');
        
        if params.addconstant == 1
            beta(:,q) = [mu_b(1,q); Hinv*mu_b(1+params.addconstant:end,q)];
            beta_mat(:,:,q) = [repmat(beta(1,q),T,1), reshape(beta(1+params.addconstant:end,q),p,T)'];       
        else
            beta(:,q) = Hinv*mu_b(1+params.addconstant:end,q);
            beta_mat(:,:,q) = reshape(beta(1+params.addconstant:end,q),p+1,T)';  
        end
     end        
    
    %% Update prior covariance matrix D
    for q = 1:nq
        d4(:,q) = sum(1./d3(:,q)) + 1;
        d3(:,q) = 1./(Ab.*d2(:,q)) + 0.5*(Tp+1+1)./d4(:,q);
        d2(:,q) = 1./Ddiag(:,q) + 1./(Ab.*d3(:,q));
        Ddiag(:,q) = 0.5*(mu_b(:,q).^2 + diag(Sigma_b(:,:,q))) + 1./d2(:,q);        
        Dinv(:,:,q) = diag(1./Ddiag(:,q)); 
    end
        
    %% Update z
    chi_z = (k1_sq./k2_sq + 2)./E_sig;
    M = zeros(T,nq);
    for q = 1:nq
        M(:,q) = (y - x*mu_b(:,q)).^2 + diag(x*Sigma_b(:,:,q)*x');
    end
    psi_z = (M./(k2_sq.*E_sig));
    E_z   = (sqrt(psi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(chi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1));
    E_iz  = (sqrt(chi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(psi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1)) - 1./psi_z;
    
    %% Update sigma2
    r_sig = r0 + 3*T;
    s_sig = s0 + sum((E_iz.*M)./(2*k2_sq) - (k1.*(y - x*mu_b)./k2_sq) + (1 + k1_sq./(2*k2_sq)).*E_z); %(1./k2_sq).*sum(k1_sq.*E_z - 2*k1.*(y - X*mu_b) +  E_iz.*M) + 2*sum(E_z); 
    E_sig = (s_sig./r_sig);    

    kappa = kappa + 1;
    
    % Get ELBO and go to the next iteration
    F_old = F_new;
    F_new = beta(:,:,1);
    
end
fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
%beta  = mu_b;
sigma = E_sig;
end        
