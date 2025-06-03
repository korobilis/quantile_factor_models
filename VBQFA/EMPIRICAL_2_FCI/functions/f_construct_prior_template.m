function [prior] = f_construct_prior_template(params, p)

    % prompt warning if defaults are used
    if params.verbose == 0
        if isfield(params, 'priors')==0
           disp('No prior set! Reverting to default.') 
        end
    end

    if params.FP_type == "OLS" 
        prior = [];
        
    % construct the prior and set defaults
    elseif params.FP_type == "Forgetting Factor"
        prior.lambda = 0.94;
        prior.kappa  = 0.94;

        % Initial condition for Kalman filter b(0) ~ N(b0,P0)
        prior.b0 = 0.5;
        prior.P0 = 10;
        prior.s0 = 0.1;
        
    elseif params.FP_type == "TVP Lasso"
        prior.CV = 5;

    elseif params.FP_type == "Variational Bayes"
        prior.q       = 0;
        prior.maxiter = 100;
        prior.prior   = 2;
        prior.delta   = 0.8;
        prior.c0      = 25;
        prior.h0      = 12;
        prior.g0      = 1;
    end

end

