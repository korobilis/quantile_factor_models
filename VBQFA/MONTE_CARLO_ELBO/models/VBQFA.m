function [F,L,ELBOsave,kappasave,mu_fsave] = VBQFA(x,r,maxiter,quant,cons,prior)

%% VBQFA Variational Bayes Quantile Factor Analysis model estimation
% =========================================================================
% INPUT:
%     x     (n x p) variable of measurements
%     r     Number of factors
%  maxiter  Maximum number of iterations for variational Bayes
%   quant   Vector of quantiles (a monotonic grid of values, between 0 and 1)
%   cons    Indicator variable for whether to include a constant 
%           (0 w/o constant, 1 with constant). Default = 0.
%   prior   Indicator variables (1 for Normal prior, 2 for ML prior).
%           Default = 1.
%  
% OUTPUT:
%     F    (n x r x q) matrix of r factors per quantile level q
%     L    (p x r x q) loadings matrices of size p x r per quantile level q
% =========================================================================
% Written by Dimitris Korobilis and Maximilian Schroeder
% University of Glasgow and Norwegian BI Business School
% This version: November 2022

if isempty(cons)
    cons = 0;
end

if isempty(prior)
    prior = 1;
end

[n,p] = size(x);

% Define median quantile to evaluate ELBO
qeval = find((quant.*100)./100 == 0.5);
if isempty(qeval); qeval = quant(1); end

% ==============| Define quantiles
nq = length(quant);
k2_sq = 2./(quant.*(1-quant));   
k1 = (1-2*quant)./(quant.*(1-quant));
k1_sq = k1.^2;

% ==============| Define priors
% lambda_j ~ N(0,D)
Ddiag = 10*ones(p,cons+r,nq); 
Dinv = repmat(0.1*eye(cons+r),1,1,p,nq);
% sigma ~ InvGam(r0/2,s0/2)
r0 = 1.0e-4;
s0 = 1.0e-4;
% machine learning prior
a = 1.0e-4;
b = 1.0e-4; 

% Initialize parameters
mu_f      = repmat(extract(zscore(x),r),1,1,nq);
Sigma_f   = zeros(r,r,n,nq);
mu_l      = zeros(cons+r,p,nq);
Sigma_l   = zeros(cons+r,cons+r,p,nq);
M         = zeros(p,n,nq);
r_sig     = ones(p,1);
s_sig     = ones(p,1);
E_sig     = ones(p,nq);
chi_z     = zeros(p,nq);
psi_z     = zeros(n,nq,p);
E_z       = ones(p,n,nq);
E_iz      = ones(p,n,nq);

% EM algorithm settings
Threshold = 1.0e-2;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

%% Start Variational Bayes iterations
ELBOsave  = [];
kappasave = [];
mu_fsave  = mu_f;
while kappa < maxiter && abs(F_new - F_old) > Threshold
    for i = 1:p
        
        mu_f_all = [ones(n,cons==1,nq) mu_f];

        for q = 1:nq
            %% Update lambda
            Sigma_l(:,:,i,q) = eye(r+cons)/((mu_f_all(:,:,q)'*diag(squeeze(E_iz(i,:,q)))*mu_f_all(:,:,q))./(k2_sq(q)*E_sig(i,q)) + squeeze(Dinv(:,:,i,q)));
            mu_l(:,i,q) = Sigma_l(:,:,i,q)*((mu_f_all(:,:,q)'*diag(squeeze(E_iz(i,:,q)))*x(:,i))./(k2_sq(q)*E_sig(i,q)) - (k1(q)/(k2_sq(q)*E_sig(i,q))).*sum(mu_f_all(:,:,q))');       
            
            %% Update Ddiag
            if prior == 2
                a_alpha = a + 0.5;
                b_alpha = b + 0.5*(mu_l(:,i,q).^2 + diag(Sigma_l(:,:,i,q)));
                Ddiag(i,:,q)   = b_alpha./a_alpha;
                Dinv(:,:,i,q) = diag(1./(Ddiag(i,:,q) + 1e-10));
            end
        end
   
        %% Update z
        chi_z(i,:) = (k1_sq./k2_sq + 2)./E_sig(i,:);        
        for q = 1:nq
            M(i,:,q) = (x(:,i) - mu_f_all(:,:,q)*mu_l(:,i,q)).^2 + diag(mu_f_all(:,:,q)*Sigma_l(:,:,i,q)*mu_f_all(:,:,q)');
        end
        psi_z(:,:,i) = (squeeze(M(i,:,:))./(k2_sq.*E_sig(i,:)));
        E_z(i,:,:)   = (sqrt(psi_z(:,:,i)).*besselk(3/2,sqrt(psi_z(:,:,i).*chi_z(i,:)),1))./(sqrt(chi_z(i,:)).*besselk(1/2,sqrt(psi_z(:,:,i).*chi_z(i,:)),1));
        E_iz(i,:,:)  = (sqrt(chi_z(i,:)).*besselk(3/2,sqrt(psi_z(:,:,i).*chi_z(i,:)),1))./(sqrt(psi_z(:,:,i)).*besselk(1/2,sqrt(psi_z(:,:,i).*chi_z(i,:)),1)) - 1./psi_z(:,:,i);
    
        %% Update sigma2
        for q = 1:nq       
            r_sig(i,q) = r0 + 3*n;
            s_sig(i,q) = s0 + sum((E_iz(i,:,q)'.*M(i,:,q)')./(2*k2_sq(q)) - (k1(q).*(x(:,i) - mu_f_all(:,:,q)*mu_l(:,i,q))./k2_sq(q)) + (1 + k1_sq(q)./(2*k2_sq(q))).*E_z(i,:,q)'); %(1./k2_sq).*sum(k1_sq.*E_z - 2*k1.*(y - x*mu_b) +  E_iz.*M) + 2*sum(E_z); %
            E_sig(i,q) = abs(s_sig(i,q)./r_sig(i,q));
        end
    end
    
    for q = 1:nq          
        %% Update factor
        if cons == 1
            mu_con = mu_l(cons,:,q);
        else
            mu_con = 0*mu_l(1,:,q);
        end
        
        for i = 1:n
            Sigma_f(:,:,i,q) = eye(r)/((mu_l(cons+1:end,:,q)*diag(squeeze(E_iz(:,i,q)./E_sig(:,q)))*mu_l(cons+1:end,:,q)')./k2_sq(q) + eye(r));
            mu_f(i,:,q) = Sigma_f(:,:,i,q)*((mu_l(cons+1:end,:,q)*diag(squeeze(E_iz(:,i,q)./E_sig(:,q)))*(x(i,:)'-mu_con'))./k2_sq(q) - (k1(q)/k2_sq(q)).*sum(diag(1./E_sig(:,q))*mu_l(cons+1:end,:,q)')' );
        end
    end

    % Get ELBO and go to the next iteration
    %%% ====| Compute the ELBO
    ELBO = 0;
    for q = qeval
        A = 0; B = 0; C = 0; D = 0; 
        for i = 1:p
            A = A + (r_sig(i,q)+2)/2*(psi(r_sig(i,q)/2)-log(s_sig(i,q)/2))-(E_sig(i,q)/(2*k2_sq(q)))*(sum(k1_sq(q).*E_z(i,:,q)'-2*k1(q).*(x(:,i) - mu_f_all(:,:,q)*mu_l(:,i,q)) + E_iz(i,:,q)'.*M(i,:,q)'));   
            B = B - E_sig(i,q)*sum(E_z(i,:,q)) - s_sig(i,q)/2*E_sig(i,q) - sum((mu_l(:,i,q)+diag(Sigma_l(:,:,i,q)))'./Ddiag(i,:,q) + 0.5*log(Ddiag(i,:,q)));   
            C = C + 0.5*log(det(Sigma_l(:,:,i,q))) + 0.5*r_sig(i,q) + log(0.5*s_sig(i,q)) + gammaln(0.5*r_sig(i,q)) -(1 + 0.5*r_sig(i,q))*psi(0.5*r_sig(i,q));
            D = D + sum(-0.25*(log(chi_z(i,q))-log(psi_z(:,q,i))) + log(2*besselk(1/2,sqrt(psi_z(:,q,i).*chi_z(i,q)),1)) + 0.5*(psi_z(:,q,i).*E_iz(i,:,q)' + chi_z(i,q).*E_z(i,:,q)'));
        end
        detsigf = 0;
        for i = 1:n
            detsigf = detsigf + 0.5*log(det(Sigma_f(:,:,i,q)));
        end
        ELBO  = detsigf + A + B + C + D;
    end
    ELBOsave = [ELBOsave; ELBO];
    kappasave = [kappasave; kappa];
    mu_fsave(:,:,:,kappa) = mu_f;

    F_old = F_new;
    F_new = 100*randn;%ELBO;
    kappa = kappa + 1;       
end
F = mu_f;
L = mu_l;
end
