function [beta_mat] = TVPQR(y,X,quant,tvp_reg)

[n,p] = size(X);
nq = length(quant);

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


beta      = rand(np,nq);
beta_mat  = zeros(n,p,nq);
mu_b      = zeros(np,nq);
for q = 1:nq
    % Run quantile regression
    mu_b(:,q) = rq_fnm(x, y, quant(q));    

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