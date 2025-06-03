function [beta_mat,sigma,Ddiag,E_z] = VBTVPQR(y,X,maxiter,quant,tvp_reg,prior,gamma_)

[n,p] = size(X);

if ~exist('gamma_','var')
    gamma_=10;
end

% ==============| create matrices useful for TVP estimation  
if tvp_reg == 0 % Constant parameter regression case
    x = X;
    np = p;
elseif tvp_reg == 1 % TVP regression case
    np = n*p;   
    H = speye(np,np) - sparse(p+1:np,1:(n-1)*p,ones(1,(n-1)*p),np,np);
    Hinv = speye(np,np)/H;
    x = full(SURform(X)*Hinv);
elseif tvp_reg == 2 % TVP only in intercept case, predictor coeffs are constant
    np = n+(p-1);
    x = [tril(ones(n)), X(:,2:end)];
elseif tvp_reg == 3 % TVP in predictor coeffs, intercept is constant
    np = n*(p-1);
    H = speye(np,np) - sparse(p:np,1:(n-1)*(p-1),ones(1,(n-1)*(p-1)),np,np);
    Hinv = speye(np,np)/H;
    bigG = full(SURform(X(:,2:end))*Hinv);    
    x = [ones(n,1) bigG];
    np = np+1;    
end

% ==============| Define quantiles
nq = length(quant);
k2_sq = 2./(quant.*(1-quant));   
k1 = (1-2*quant)./(quant.*(1-quant));
k1_sq = k1.^2;

% ==============| Define priors
% beta_j ~ N(0,D)
Ddiag = ones(np,nq); 
Dinv = repmat(eye(np),1,1,nq);
% D ~ Horseshoe+
Ab = gamma_;%((gamma./(np+1))).^2;
% sigma ~ InvGam(r0/2,s0/2)
r0 = .01;
s0 = .01;
% Sparse Bayesian learning prior
a = 1.0e-4;
b = 1.0e-4; 

% Initialize parameters
beta      = rand(np,nq);
beta_mat  = zeros(n,p,nq);
mu_b      = zeros(np,nq);
Sigma_b   = zeros(np,np,nq);
E_sig     = ones(1,nq);
E_z       = ones(n,nq);
E_iz      = ones(n,nq);
d4        = ones(1,nq);
d3        = ones(np,nq);   
d2        = ones(np,nq);

% EM algorithm settings
Threshold = 1.0e-6;
F_new     = -1000;
F_old     = 1;
kappa     = 1;

%% Start Variational Bayes iterations
%format bank;
%fprintf('\n')
ELL = [];
while kappa < maxiter && max(abs((F_new - F_old))) > Threshold
    % Print iterations   
    %if mod(kappa,25) == 1        
    %   fprintf('%4f   %4f \n',norm((F_new - F_old)),kappa)        
    %end
    
    %% Update beta
    if np>n
        u = sqrt(E_iz./(E_sig.*k2_sq));
        y_tilde = (y.*u - k1./(E_sig.*k2_sq));
        for q = 1:nq
            xnew = x.*u(:,q);
            D = diag(Ddiag(:,q));
            U = bsxfun(@times,Ddiag(:,q),xnew');
            Sigma_b(:,:,q) = D - (U/(eye(n) + xnew*U))*U';
            mu_b(:,q) = Sigma_b(:,:,q)*(xnew'*y_tilde(:,q));
               
            % convert vector of parameters into a matrix (and accumulate coefficients if using TVP)
            if tvp_reg == 0
                beta(:,q) = mu_b(:,q);                      % In the constant parameter case, betaD are the beta coefficients we need
                beta_mat(:,:,q) = repmat(mu_b(:,q)',n,1);
            elseif tvp_reg == 1
                beta(:,q) = Hinv*mu_b(:,q);                 % betaD are TVPs in differences (beta_{t} - beta_{t-1}), convert to vector of TVPs in levels (beta_{t})
                beta_mat(:,:,q) = reshape(beta(:,q),p,n)';  % beta_mat is T x (p+1) x n_q matrix of TVPs
            elseif tvp_reg == 2
                beta(:,q) = [cumsum(mu_b(1:n,q)); mu_b(n+1:end,q)];
                beta_mat(:,:,q) = [beta(1:n,q) , repmat(beta(n+1:end,q)',n,1)];
            elseif tvp_reg == 3
                beta(:,q) = [mu_b(1,q); Hinv*mu_b(2:end,q)];
                beta_mat(:,:,q) = [repmat(beta(1,q),n,1) , reshape(beta(2:end,q),p-1,n)'];
            end              
        end
    elseif np<=n
        for q = 1:nq
            Sigma_b(:,:,q) = eye(np)/((x'*diag(E_iz(:,q))*x)./(k2_sq(q)*E_sig(q)) + Dinv(:,:,q));
            mu_b(:,q) = Sigma_b(:,:,q)*((x'*diag(E_iz(:,q))*y)./(k2_sq(q)*E_sig(q)) - (k1(q)/(k2_sq(q)*E_sig(q))).*sum(x)');
            
            % convert vector of parameters into a matrix (and accumulate coefficients if using TVP)
            if tvp_reg == 1
                beta(:,q) = Hinv*mu_b(:,q);                 % betaD are TVPs in differences (beta_{t} - beta_{t-1}), convert to vector of TVPs in levels (beta_{t})
                beta_mat(:,:,q) = reshape(beta(:,q),p,n)';  % beta_mat is T x (p+1) x n_q matrix of TVPs
            elseif tvp_reg == 2
                beta(:,q) = [cumsum(mu_b(1:n,q)); mu_b(n+1:end,q)];
                beta_mat(:,:,q) = [beta(1:n,q) , repmat(beta(n+1:end,q)',n,1)];
            elseif tvp_reg == 0
                beta(:,q) = mu_b(:,q);                      % In the constant parameter case, betaD are the beta coefficients we need
                beta_mat(:,:,q) = repmat(mu_b(:,q)',n,1);
            elseif tvp_reg == 3
                beta(:,q) = [mu_b(1,q); Hinv*mu_b(2:end,q)];
                beta_mat(:,:,q) = [repmat(beta(1,q),n,1) , reshape(beta(2:end,q),p-1,n)']; 
            end              
        end        
    end     
    
    %% Update prior covariance matrix D
    for q = 1:nq
        if prior == 0
            Ddiag(:,q) = 1e+10*ones(np,1);
            if np<=n
                Dinv(:,:,q) = diag(1./Ddiag(:,q)); 
            end
        elseif prior == 1
            d4(:,q) = sum(1./d3(:,q)) + 1;
            d3(:,q) = 1./(Ab.*d2(:,q)) + 0.5*(np+1)./d4(:,q);
            d2(:,q) = 1./Ddiag(:,q) + 1./(Ab.*d3(:,q));
            Ddiag(:,q) = min(1e+5,0.5*(mu_b(:,q).^2 + diag(Sigma_b(:,:,q))) + 1./d2(:,q));
            Ddiag(1,q) = 100;
            if np<n
                Dinv(:,:,q) = diag(1./Ddiag(:,q)); 
            end
        elseif prior == 2
            Ddiag(:,q) = (b + mu_b(:,q).^2 + diag(Sigma_b(:,:,q)))/(a);            
            if np<=n
                Dinv(:,:,q) = diag(1./Ddiag(:,q)); 
            end
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
    s_sig = s0 + sum((E_iz.*M)./(2*k2_sq) - (k1.*(y - x*mu_b)./k2_sq) + (1 + k1_sq./(2*k2_sq)).*E_z); %(1./k2_sq).*sum(k1_sq.*E_z - 2*k1.*(y - x*mu_b) +  E_iz.*M) + 2*sum(E_z); %
    E_sig = (s_sig./r_sig); % ones(1,nq);%(s_sig./r_sig); %

    kappa = kappa + 1;
    
%     % Get ELBO and go to the next iteration
%     %%% ====| Compute the ELBO
%     %(sum(k1_sq.*E_z-2*k1.*(y - x*mu_b) + E_iz.*(M))
%     L = NaN(1,nq);
%     H = NaN(1,nq);
%     for q = 1:nq
%         a = (r_sig+2)/2*(psi(r_sig/2)-log(s_sig(:,q)/2))-(E_sig(:,q)/(2*k2_sq(:,q)))*(sum(k1_sq(:,q).*E_z(:,q)-2*k1(:,q).*(y - x*mu_b(:,q)) + E_iz(:,q).*(M(:,q))));
%         b = -E_sig(:,q)*sum(E_z(:,q)) - s_sig(:,q)/2*E_sig(:,q) - sum((mu_b(:,q)+diag(Sigma_b(:,:,q)))./Ddiag(:,q) + 0.5*log(Ddiag(:,q)));
%         c = -3/2*(sum(log(Ddiag(:,q).*d2(:,q).*d3(:,q)))) - log(d4(:,q)) - ((p+1)/(2.*d4(:,q))); 
%         L(:,q) = a+b+c;
% 
%         Ha = 0.5*log(det(Sigma_b(:,:,q))) + 0.5*r_sig+log(0.5*s_sig(:,q)) + gammaln(0.5*r_sig) -(1+0.5*r_sig)*psi(0.5*r_sig); % issue with r_sig causing inf or perhaps gamma function does not make sense?
%         Hb = sum(-0.25*(log(chi_z(:,q))-log(psi_z(:,q))) + log(2*besselk(1/2,sqrt(psi_z(:,q).*chi_z(:,q)),1)) +0.5*(psi_z(:,q).*E_iz(:,q)+chi_z(:,q).*E_z(:,q))) + 1/2*(sum(log(Ddiag(:,q).*d2(:,q).*d3(:,q)))) - log(d4(:,q));
%         H(:,q) = Ha+Hb;
% 
%         ELBO = L+H;
%     end 
%     ELL = [ELL; ELBO];
%     
%     F_old = F_new;
%     F_new = ELBO; %beta_mat(:);
    
end
%fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c',8,8,8,8,8,8,8,8,8,8,8,8,8,8)
%beta  = mu_b;
sigma = E_sig;
end
