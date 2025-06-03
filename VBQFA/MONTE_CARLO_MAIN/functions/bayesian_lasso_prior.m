function [Q,lam,tau] = bayesian_lasso_prior(beta,sigma2,tau,kappa,p)

    lam  = gamrnd(p + 1,(0.5*sum(tau) + kappa));		
    tau  = min(1./random('InverseGaussian',sqrt((lam*sigma2)./(beta'.^2)),lam,p,1),1e+6);
    Q    = tau;
end