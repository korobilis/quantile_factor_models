function [Q,gamma,tau,pi0,kappa,lambda,xi,nu] = sns_prior(y,X,sigma2,beta,gamma,tau,pi0,kappa,lambda,xi,nu)

p = length(beta);
sigma = sqrt(sigma2);

% SSVS/SNS class of priors
% 1) Update model probabilities
for j = randperm(p)
    theta = beta.*gamma;
    theta_star = theta; 
    theta_star_star = theta;
    theta_star(j) = beta(j); 
    theta_star_star(j) = 0;
    l_0       = lnormpdf(0, sum((y - X*theta_star_star).^2), sigma);
    l_1       = lnormpdf(0, sum((y - X*theta_star).^2), sigma);
    pip       = 1./( 1 + ((1-pi0).*pi0).* exp(l_0 - l_1) );
    gamma(j,1)= binornd(1,pip);
end

pi0 = betarnd(1 + sum(gamma==1),1 + sum(gamma==0));


Q = tau;     
        
end

