function [F,L] = SVBQFA(x,r,maxiter,quant,cons,prior,batch_)

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

if isempty(batch_)
    batch_ = 1;
end

[n,p] = size(x);

% Define median quantile to evaluate ELBO
qeval = find((quant.*100)./100 == 0.5);
if isempty(qeval) 
    qeval = quant(1); 
end

% ==============| Define quantiles
nq = length(quant);
k2_sq = 2./(quant.*(1-quant));   
k1 = (1-2*quant)./(quant.*(1-quant));
k1_sq = k1.^2;

% ==============| Define priors
% lambda_j ~ N(0,D)
Ddiag    = 10*ones(p,cons+r,nq); 
Dinv     = repmat(0.1*eye(cons+r),1,1,p,nq);

% sigma ~ InvGam(r0/2,s0/2)
r0 = 1.0e-4;
s0 = 1.0e-4;
% machine learning prior
a = 1.0e-4;
b = 1.0e-4; 

% Initialize parameters
[fac,lam] = extract(zscore(x),r);
mu_f      = repmat(fac,1,1,nq);
Sigma_f   = zeros(r,r,nq);
mu_l      = repmat(lam',1,1,nq);
Sigma_l   = zeros(cons+r,cons+r,p,nq);
E_sig     = ones(p,nq);
E_z       = ones(p,n,nq);
E_iz      = ones(p,n,nq);

% EM algorithm settings
Threshold = 1.0e-6;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

kappa_ = 0.9;
tau_   = 1;

%% Start Variational Bayes iterations
while kappa < maxiter && abs(F_new - F_old) > Threshold
    
    rho_ = (kappa+tau_)^(-kappa_);
    %% update random local parameter, i.e. states
    
    % Sample random datapoint
    ids = randsample(n,batch_);
    
    for q = 1:nq          
        %% Update factor
        if cons == 1
            mu_con = mu_l(cons,:,q);
        else
            mu_con = 0*mu_l(1,:,q);
        end
        
        for id = 1:length(ids)
            i = ids(id);
            Sigma_f(:,:,q) = eye(r)/((mu_l(cons+1:end,:,q)*diag(squeeze(E_iz(:,i,q)./E_sig(:,q)))*mu_l(cons+1:end,:,q)')./k2_sq(q) + eye(r));
            mu_f(i,:,q) = Sigma_f(:,:,q)*((mu_l(cons+1:end,:,q)*diag(squeeze(E_iz(:,i,q)./E_sig(:,q)))*(x(i,:)'-mu_con'))./k2_sq(q) - (k1(q)/k2_sq(q)).*sum(diag(1./E_sig(:,q))*mu_l(cons+1:end,:,q)')' );
        end
    end

    %% update global parameter at random data point
    
    mu_l_int  = repmat(zeros(size(lam')),1,1,nq);
    E_sig_int = zeros(p,nq);
    E_z_int   = zeros(p,n,nq);
    E_iz_int  = zeros(p,n,nq);
    Dinv_int = repmat(zeros(cons+r),1,1,p,nq);

    for id = 1:length(ids)
        
        ii = ids(id);
        x_aux    = ones(size(x)).*x(ii,:);
        mu_f_aux = repmat(mu_f(ii,:,:),n,1,1);
        mu_f_all = [ones(n,cons==1,nq) mu_f_aux];
        
        for i = 1:p

           for q = 1:nq
               %% Update lambda
               Sigma_l(:,:,i,q) = eye(r+cons)/((mu_f_all(:,:,q)'*diag(squeeze(E_iz(i,:,q)))*mu_f_all(:,:,q))./(k2_sq(q)*E_sig(i,q)) + squeeze(Dinv(:,:,i,q)));
               mu_l_int(:,i,q) = mu_l_int(:,i,q) + Sigma_l(:,:,i,q)*((mu_f_all(:,:,q)'*diag(squeeze(E_iz(i,:,q)))*x_aux(:,i))./(k2_sq(q)*E_sig(i,q)) - (k1(q)/(k2_sq(q)*E_sig(i,q))).*sum(mu_f_all(:,:,q))');       

               %% Update Ddiag
               if prior == 2
                   a_alpha = a + 0.5;
                   b_alpha = b + 0.5*(mu_l(:,i,q).^2 + diag(Sigma_l(:,:,i,q)));
                   Ddiag   = b_alpha./a_alpha;
                   Dinv_int(:,:,i,q) = Dinv_int(:,:,i,q) + diag(1./(Ddiag + 1e-10));
               end
           end

           %% Update z
           chi_z = (k1_sq./k2_sq + 2)./E_sig(i,:);
           M = zeros(n,nq);
           for q = 1:nq
               M(:,q) = (x_aux(:,i) - mu_f_all(:,:,q)*mu_l(:,i,q)).^2 + diag(mu_f_all(:,:,q)*Sigma_l(:,:,i,q)*mu_f_all(:,:,q)');
           end
           psi_z = (M./(k2_sq.*E_sig(i,:)));
           E_z_int(i,:,:)   =  E_z_int(i,:,:) + reshape((sqrt(psi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(chi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1)),1,n,nq);
           E_iz_int(i,:,:)  =  E_iz_int(i,:,:) + reshape((sqrt(chi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(psi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1)) - 1./psi_z,1,n,nq);

           %% Update sigma2
           r_sig = r0 + 3*n;
           s_sig = s0 + sum((squeeze(E_iz(i,:,:)).*M)./(2*k2_sq) - (k1.*(x_aux(:,i) - mu_f_all(:,:,q)*mu_l(:,i,q))./k2_sq) + (1 + k1_sq./(2*k2_sq)).*squeeze(E_z(i,:,:))); %(1./k2_sq).*sum(k1_sq.*E_z - 2*k1.*(y - x*mu_b) +  E_iz.*M) + 2*sum(E_z); %
           E_sig_int(i,:) = E_sig_int(i,:) + abs(s_sig./r_sig);
        end
        
        if id == length(ids)
            mu_l  = (1-rho_).*mu_l + (rho_/batch_).*mu_l_int;
            E_sig = (1-rho_).*E_sig + (rho_/batch_).*E_sig_int;
            E_z   = (1-rho_).*E_z + (rho_/batch_).*E_z_int;
            E_iz  = (1-rho_).*E_iz + (rho_/batch_).*E_iz_int;
            Dinv  = (1-rho_).*Dinv + (rho_/batch_).*Dinv_int;
        end
    end
    
    
    
%     
%     for i = 1:p
%         
%         mu_f_all = [ones(n,cons==1,nq) mu_f];
% 
%         for q = 1:nq
%             %% Update lambda
%             Sigma_l(:,:,i,q) = eye(r+cons)/((mu_f_all(:,:,q)'*diag(squeeze(E_iz(i,:,q)))*mu_f_all(:,:,q))./(k2_sq(q)*E_sig(i,q)) + squeeze(Dinv(:,:,i,q)));
%             mu_l(:,i,q) = Sigma_l(:,:,i,q)*((mu_f_all(:,:,q)'*diag(squeeze(E_iz(i,:,q)))*x(:,i))./(k2_sq(q)*E_sig(i,q)) - (k1(q)/(k2_sq(q)*E_sig(i,q))).*sum(mu_f_all(:,:,q))');       
%             
%             %% Update Ddiag
%             if prior == 2
%                 a_alpha = a + 0.5;
%                 b_alpha = b + 0.5*(mu_l(:,i,q).^2 + diag(Sigma_l(:,:,i,q)));
%                 Ddiag   = b_alpha./a_alpha;
%                 Dinv(:,:,i,q) = diag(1./(Ddiag + 1e-10));
%             end
%         end
%    
%         %% Update z
%         chi_z = (k1_sq./k2_sq + 2)./E_sig(i,:);
%         M = zeros(n,nq);
%         for q = 1:nq
%             M(:,q) = (x(:,i) - mu_f_all(:,:,q)*mu_l(:,i,q)).^2 + diag(mu_f_all(:,:,q)*Sigma_l(:,:,i,q)*mu_f_all(:,:,q)');
%         end
%         psi_z = (M./(k2_sq.*E_sig(i,:)));
%         E_z(i,:,:)   = (sqrt(psi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(chi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1));
%         E_iz(i,:,:)  = (sqrt(chi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(psi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1)) - 1./psi_z;
%     
%         %% Update sigma2
%         r_sig = r0 + 3*n;
%         s_sig = s0 + sum((squeeze(E_iz(i,:,:)).*M)./(2*k2_sq) - (k1.*(x(:,i) - mu_f_all(:,:,q)*mu_l(:,i,q))./k2_sq) + (1 + k1_sq./(2*k2_sq)).*squeeze(E_z(i,:,:))); %(1./k2_sq).*sum(k1_sq.*E_z - 2*k1.*(y - x*mu_b) +  E_iz.*M) + 2*sum(E_z); %
%         E_sig(i,:) = abs(s_sig./r_sig);
%     end
%     
%     
    % Get ELBO and go to the next iteration
    %%% ====| Compute the ELBO
    for q = qeval
        detsigl  = 0;
        for i = 1:p
            detsigl = detsigl + det(Sigma_l(:,:,i,q));
        end
        ELBO  = n*sum(log(det(Sigma_f(:,:,q)))) + p*sum(log(detsigl)) - sum(r_sig*log(s_sig(q))); 
    end

    F_old = F_new;
    F_new = ELBO;
    kappa = kappa + 1;       
end



F = mu_f;
L = mu_l;
end
