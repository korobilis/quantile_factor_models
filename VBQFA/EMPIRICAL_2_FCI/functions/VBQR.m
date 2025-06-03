function [beta,Sigma_b,sigma,Ddiag,d2,d3,d4] = VBQR(y,X,maxiter,quant)

%% Description goes here
X = [ones(size(X,1),1) X];
[n,p] = size(X);

% ==============| Define quantiles
nq = length(quant);
k2_sq = 2./(quant.*(1-quant));   
k1 = (1-2*quant)./(quant.*(1-quant));
k1_sq = k1.^2;

% ==============| Define priors
% beta_j ~ N(0,D)
Ddiag = ones(p,nq); 
if p<n
    Dinv = repmat(eye(p),1,1,nq);
end
% D ~ Horseshoe+
Ab = (0.0001)^2;
% sigma ~ InvGam(r0/2,s0/2)
r0 = .01;
s0 = .01;

% Initialize parameters
mu_b      = zeros(p,nq);
Sigma_b   = zeros(p,p,nq);
E_sig     = ones(1,nq);
E_z       = ones(n,nq);
E_iz      = ones(n,nq);
d4        = ones(1,nq);
d3        = ones(p,nq);   
d2        = ones(p,nq);

% EM algorithm settings
Threshold = 1.0e-6;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

%% Start Variational Bayes iterations
format bank;
fprintf('Now you are running VBVS')
fprintf('\n')
fprintf('Iteration 0000')
while kappa < maxiter && abs((F_new - F_old)) > Threshold
    % Print iterations   
    if mod(kappa,10) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%4d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',kappa)
    end
    
    %% Update beta
    if p>=n
        u = sqrt(E_iz./(E_sig.*k2_sq));
        y_tilde = (y.*u - k1./k2_sq);
        for q = 1:nq
            xnew = X.*u(:,q);
            D = diag(Ddiag(:,q));
            U = bsxfun(@times,Ddiag(:,q),xnew');
            Sigma_b(:,:,q) = D - (U/(eye(n) + xnew*U))*U';
            mu_b(:,q) = Sigma_b(:,:,q)*((X'*diag(E_iz(:,q))*y)./(k2_sq(q)*E_sig(q)) - (k1(q)/(k2_sq(q))).*sum(X)');%(xnew'*y_tilde(:,q));
        end
    elseif p<n
        for q = 1:nq
            Sigma_b(:,:,q) = eye(p)/((X'*diag(E_iz(:,q))*X)./(k2_sq(q)*E_sig(q)) + Dinv(:,:,q));
            mu_b(:,q) = Sigma_b(:,:,q)*((X'*diag(E_iz(:,q))*y)./(k2_sq(q)*E_sig(q)) - (k1(q)/(k2_sq(q))).*sum(X)');
        end        
    end

    %% Update prior covariance matrix D
    for q = 1:nq
        d4(:,q) = sum(1./d3(:,q)) + 1;
        d3(:,q) = 1./(Ab.*d2(:,q)) + 0.5*(p+1)./d4(:,q);
        d2(:,q) = 1./Ddiag(:,q) + 1./(Ab.*d3(:,q));
        Ddiag(:,q) = min(1e+2,0.5*(mu_b(:,q).^2 + diag(Sigma_b(:,:,q))) + 1./d2(:,q));
        %Ddiag(1,q) = 100;
        if p<n
            Dinv(:,:,q) = diag(1./Ddiag(:,q)); 
        end
    end
        
    %% Update z
    chi_z = (k1_sq./k2_sq + 2)./E_sig;
    M = zeros(n,nq);
    for q = 1:nq
        M(:,q) = (y - X*mu_b(:,q)).^2 + diag(X*Sigma_b(:,:,q)*X');
    end
    psi_z = (M./(k2_sq.*E_sig));
    E_z   = (sqrt(psi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(chi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1));
    E_iz  = (sqrt(chi_z).*besselk(3/2,sqrt(psi_z.*chi_z),1))./(sqrt(psi_z).*besselk(1/2,sqrt(psi_z.*chi_z),1)) - 1./psi_z;
    
    %% Update sigma2
    r_sig = r0 + 3*n;
    s_sig = s0 + (1./k2_sq).*sum(k1_sq.*E_z - 2*k1.*(y - X*mu_b) +  E_iz.*M) + 2*sum(E_z); %sum((E_iz.*M)./(2*k2_sq) - (k1.*(y - X*mu_b)./k2_sq) + (1 + k1_sq./(2*k2_sq)).*E_z); %
    E_sig = (s_sig./r_sig);    

    kappa = kappa + 1;
    
    % Get ELBO and go to the next iteration
    F_old = F_new;
    F_new = rand;
    
end
fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
beta  = mu_b;
sigma = E_sig;
end
