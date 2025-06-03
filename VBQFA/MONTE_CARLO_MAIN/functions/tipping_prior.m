function [Q,lambda,tau] = tipping_prior(beta,a,b)

    lambda = a + 0.5;
    tau    = b + 0.5*(beta.^2);
    Q      = 1./gamrnd(lambda,tau);

    %% estimate of prior mean
    miu = zeros(length(beta),1);