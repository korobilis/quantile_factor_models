function [F,L] = MCMCQFA(x,r,quant,cons,nburnin,ndraws,nthin,ALsampler,prior)

%% MCMCQFA Markov chain Monte Carlo Quantile Factor Analysis model estimation
% =========================================================================
% INPUT:
%     x     (n x p) variable of measurements
%     r     Number of factors
%  maxiter  Maximum number of iterations for variational Bayes
%   quant   Vector of quantiles (a monotonic grid of values, between 0 and 1)
%   cons    Indicator variable for whether to include a constant 
%           (0 w/o constant, 1 with constant). Default = 0.
%   prior   1 = Horseshoe
%           2 = Tipping Machine Learning Prior
%           3 = Bhattacharya & Dunson (non-adaptive)
%           4 = Bhattacharya & Dunson (adaptive)
%           5 = Bayesian Lasso (Park and Casella 2008) 
%           6 = Spike and Slab
%           7 = SSVS normal
%           8 = SSVS student-t
%           9 = SSVS lasso
%          10 = SSVS lasso
%          11 = SSVS horseshoe
%          12 = Lagramanti et al. CUSP prior
%  
% OUTPUT:
%     F    (n x r x q) matrix of r factors per quantile level q
%     L    (p x r x q) loadings matrices of size p x r per quantile level q
% =========================================================================
% Written by Dimitris Korobilis and Maximilian Schroeder
% University of Glasgow and Norwegian BI Business School
% First version: November 2022
% This version: February 2024


% determine size of data
[n,p] = size(x);

if isempty(cons)
    cons = 0;
end

if isempty(prior)
    prior = 1;
end

% ==============| Define quantiles
nq = length(quant);
k2_sq = 2./(quant.*(1-quant));
k1 = (1-2*quant)./(quant.*(1-quant));
k1_sq = k1.^2;

% ==============| Define priors
% sigma ~ InvGam(as1/2,as2/2)
as1 = 0.01;
as2 = 0.01;

% prior for L
if prior == 1 % Horseshoe prior
    lambdaL   = 0.1*ones(p,r+cons,nq);     % "local" shrinkage parameters
    tauL      = 0.1*ones(p,nq);                      % "global" shrinkage parameter
    nuL       = 0.1*ones(p,r+cons,nq);  
    xiL       = 0.1*ones(p,nq);
    QL        = 1*ones(r+cons,p,nq);
elseif prior == 2 % Machine Learning prior
    lambdaL = 1.0e-4;
    tauL    = 1.0e-4; 
    QL        = 1*ones(r+cons,p,nq);
elseif prior == 3 || prior == 4
    df  = 3;
    ad1 = 2.1;
    bd1 = 1;
    ad2 = 3.1;
    bd2 = 1;

    as1 = 1;
    as2 = 0.3;
    
    psijh = repmat(gamrnd(df/2,2/df,[p,r+cons]),1,1,nq);                            % local shrinkage coefficients
    delta = repmat([gamrnd(ad1,bd1);gamrnd(ad2,bd2,[r-1+cons,1])],1,1,nq);          % gobal shrinkage coefficients multilpliers
    tauh  = cumprod(delta);                                                         % global shrinkage coefficients
    QL    = permute(bsxfun(@times,psijh,permute(tauh,[2,1,3])),[2,1,3]);            % precision of loadings rows   
    
    % if adaptive
    if prior == 4
        b0      = 1;
        b1      = 0.0005;
        epsilon = 1e-0;
        prop    = 1;
        rs      = ones(1,nq).*r;
        r_max   = 20;
        r       = r_max;
    end
elseif prior == 5
    tauL   = ones(p,r+cons,nq);
    kappaL = 3;
    QL     = 1*ones(r+cons,p,nq);
elseif prior == 6
    tauL    = 9*ones(p,r+cons,nq);
    pi0L    = 0.2*ones(p,nq);
    kappaL  = 1;
    lambdaL = ones(p,r+cons,nq);
    xiL     = 1;
    nuL     = ones(p,r+cons,nq);
    gammaL  = ones(p,r+cons,nq);
    QL     = 1*ones(r+cons,p,nq);
elseif prior == 7 || prior == 8 || prior == 9 || prior == 10 || prior == 11
    tau1L   = 1*ones(p,r+cons,nq);
    tau0L   = (1/n)*ones(p,r+cons,nq);
    pi0L    = 0.2*ones(p,nq);
    kappaL  = NaN(p,nq);
    xiL     = NaN(p,nq);
    nuL     = NaN(p,r+cons,nq);
    gammaL  = ones(p,r+cons,nq); 
    QL     = 1*ones(r+cons,p,nq);
    if prior == 7   % Simple SSVS
        method = 'ssvs_normal';
        pi0 = 0.2;
        fvalue = tpdf(sqrt(2.1*log(r+cons)),5);
        tau1L   = max(100*tau0L,pi0*tau0L/((1 - pi0)*fvalue));
    elseif prior == 8
        method = 'ssvs_student';
        kappaL  = 0.01.*ones(p,nq);
    elseif prior == 9
        method = 'ssvs_lasso';
        kappaL  = 3.*ones(p,nq);
    elseif prior == 10
        method = 'sns_lasso';
        kappaL  = 3.*ones(p,nq);
    elseif prior == 11
        method = 'ssvs_horseshoe';
        kappaL  = ones(p,nq);
        xiL     = ones(p,nq);
        nuL     = ones(p,r+cons,nq);      
    end 
elseif prior == 12
    r           = p;
    u           = rand(ndraws,nq);
    H           = NaN(1,nq);
    Hstar       = NaN(1,nq);
    stbw        = NaN(cons+r,nq);
    z_star      = NaN(r,nq);
    H(1,:)      = r;
    Hstar(1,:)  = r; 
    alpha       = 5;
    alpha0      = -1;
    alpha1      = -5*10^(-4);
    a_theta     = 2;
    b_theta     = 2;
    theta_inf   = 0.05;
    theta_inv   = ones(r,nq);
    start_adapt = 500;
    QL          = 1*ones(r+cons,p,nq);
    w_post      = 1./H.*ones(H(1),nq);
end

% Initialize parameters
F	  = repmat(extract(zscore(x),r),1,1,nq);
L     = zeros(cons+r,p,nq);
Sigma = ones(p,nq);
z     = ones(n,p,nq);
v     = ones(n,p,nq).*reshape(k2_sq,1,1,3);
FL    = zeros(n,p,nq);

% define output structures
F_draws = zeros(n,r,nq,ndraws/nthin);
L_draws = zeros(r+cons,p,nq,ndraws/nthin);
z_draws = zeros(n,p,nq,ndraws/nthin);

if prior == 12
   H_draws = zeros(ndraws/nthin,nq); 
end

%% Start Variational Bayes iterations
format bank;
fprintf('Now you are running QFAVAR with MCMC')
fprintf('\n')
fprintf('Iteration 000000')
savedraw = 0; tic;

for irep = 1:(ndraws+nburnin)
    
    % Print every "iter" iterations on the screen
    if mod(irep,100) == 0
        fprintf('%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%c%s%6d',8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,'Iteration ',irep)
    end
    
    %%%%%%%% ================================================================================================== %%%%%%%%
    %%%%%%%% =====================| MCMCQFA measurement equation (Factor extraction) |========================= %%%%%%%%
    for q = 1:nq
        if prior == 4
           r = rs(q);
        elseif prior == 12
           r = H(q);
        end
        
        F_all = [ones(n,cons==1,nq) F];
        for i = 1:p            
            select = [ones(1,cons) cons+(1:r)];
            %% =====| Step 1: Sample loadings L
            %v = sqrt(Sigma(i,q)*k2_sq(q)*z(:,i,q));                                                    % This is the variance of the Asymetric Laplace errors
            x_tilde = (x(:,i) - k1(q).*z(:,i,q))./v(:,i,q);                                             % Standardized LHS variables
            F_tilde = F_all(:,select,q)./v(:,i,q);                                                      % Standardized RHS variables                     
            L(select,i,q) = randn_gibbs(x_tilde,F_tilde,QL(select,i,q),n,length(select),1);             % Sample loadings Lambda
            if prior == 1
                [QL(select,i,q),~,~,lambdaL(i,select,q),tauL(i,q),nuL(i,select,q),xiL(i,q)] = ...
                horseshoe_prior(L(select,i,q)',length(select),tauL(i,q),nuL(i,select,q),xiL(i,q));      % sample prior variance of loadings Lambda
            elseif prior == 2
                [QL(select,i,q),~,~] = tipping_prior(L(select,i,q)',lambdaL,tauL);
            elseif prior == 5
                [QL(select,i,q),~,~] = bayesian_lasso_prior(L(select,i,q)',1,tauL(i,select,q),kappaL,length(select));
            elseif prior == 6
                [QL(select,i,q),gammaL(i,select,q),tauL(i,select,q),pi0L(i,q),kappaL,lambdaL(i,select,q),xiL,nuL(i,select,q)] = ...
                    sns_prior(x_tilde,F_tilde,1,L(select,i,q),gammaL(i,select,q)',tauL(i,select,q),pi0L(i,q),kappaL,lambdaL(i,select,q),xiL,nuL(i,select,q));
            elseif prior == 7 || prior == 8 || prior == 9 || prior == 10 || prior == 11
                [QL(select,i,q),gammaL(i,select,q),tau0L(i,select,q),tau1L(i,select,q),pi0L(i,q),kappaL(i,q),xiL(i,q),nuL(i,select,q)] = ssvs_prior(L(select,i,q)',1,tau0L(i,select,q),tau1L(i,select,q),pi0L(i,q),kappaL(i,q),xiL(i,q),nuL(i,select,q),method);
            end
        
      
            %% =====| Step 2: Sample latent indicators z    
            FL(:,i,q) = F_all(:,:,q)*L(:,i,q);
            if ALsampler == 1             % Khare and Robert (2012)                          
                chi_z = sqrt(k1_sq(q) + 2*k2_sq(q))./abs(x(:,i) - squeeze(FL(:,i,q)));      % posterior moment k1 of z
                psi_z = (k1_sq(q) + 2*k2_sq(q))./(Sigma(i,q).*k2_sq(q));                    % posterior moment k2 of z
                z(:,i,q)  = min(1./random('InverseGaussian',chi_z,psi_z,n,1),1e+6);         % Sample z from Inverse Gaussian
            elseif ALsampler == 2         % Kozumi and Kobayashi (2011)
                chi_z = ((x(:,i) - squeeze(FL(:,i,q))).^2)./(Sigma(i,q).*k2_sq(q));         % posterior moment k1 of z
                psi_z = (k1_sq(q) + 2*k2_sq(q))./(Sigma(i,q).*k2_sq(q));                    % posterior moment k2 of z
                for t = 1:n
                    z(t,i,q)  = min(gigrnd(0.5,psi_z,chi_z(t),1),1e+6);                     % Sample z from Generalized Inverse Gaussian
                end
            end

            %% =====| Step 3: Sample factor regression variances Sigma
            a1 = as1 + 3*n/2;      
            sse = (x(:,i) - FL(:,i,q) - k1(q).*z(:,i,q)).^2;
            a2 = as2 + sum(sse./(2*z(:,i,q).*k2_sq(q))) + sum(z(:,i,q));       
            Sigma(i,q) = 1./gamrnd(a1,1./a2);                                              % Sample Sigma from inverse-Gamma
            
            v(:,i,q) = sqrt(Sigma(i,q)*k2_sq(q)*z(:,i,q));
        end
        
        if prior == 3
            [Q,tauh(:,:,q),delta(:,:,q),psijh(:,:,q)]       = Bhattacharya_Dunson_prior(L(:,:,q)',tauh(:,:,q),delta(:,:,q),df,ad1,bd1,ad2,bd2,p,r+cons);
            QL(:,:,q) = Q'; 
        end
        
        if prior == 4
            [Q,tauh(1:r+cons,:,q),delta(1:r+cons,:,q),psijh(:,1:r+cons,q)] = Bhattacharya_Dunson_prior(L(1:r+cons,:,q)',tauh(1:r+cons,:,q),delta(1:r+cons,:,q),df,ad1,bd1,ad2,bd2,p,r+cons);
            QL(1:r+cons,:,q) = Q'; 
            
            
            prob = 1/exp(b0 + b1*irep);                % probability of adapting
            uu = rand;
            lind = sum(abs(L(1:r+cons,:,q)') < epsilon)/p;    % proportion of elements in each column less than eps in magnitude
            vec = lind >=prop;
            num = sum(vec);       % number of redundant columns

            if uu < prob
                if  i > 20 && num == 0 && all(lind < 0.995)
                    if r+cons<r_max
                        r = r + 1;
                    end
                    L(r+cons,:,q) = zeros(1,p);
                    F(:,r+cons,q) = normrnd(0,1,[n,1]);
                    psijh(:,r+cons,q) = gamrnd(df/2,2/df,[p,1]);
                    delta(r+cons,:,q) = gamrnd(ad2,1/bd2);
                    tauh(1:r+cons,:,q)  = cumprod(delta(1:r+cons,:,q));
                    Q = bsxfun(@times,psijh(:,:,q),tauh(:,:,q)');
                    QL(1:r+cons,:,q) = Q(:,1:r+cons)'; 
                elseif num > 0
                    nonred = setdiff(1:r+cons,find(vec)); % non-redundant loadings columns
                    r = max(r - num,1);
                    L(1:r+cons,:,q) = L(nonred,:,q);
                    L(r+cons+1:end,:,q) = 0; 
                    psijh(:,1:r+cons,q) = psijh(:,nonred,q);
                    psijh(:,r+cons+1:end,q) = 0;
                    F(:,1:r+cons,q) = F(:,nonred,q); 
                    F(:,r+cons+1:end,q) = 0; 
                    delta(1:r+cons,:,q) = delta(nonred,:,q);
                    delta(r+cons+1:end,:,q) = 0;
                    tauh(1:r+cons,:,q) = cumprod(delta(1:r+cons,:,q));
                    tauh(r+cons+1:end,:,q) = 0;
                    Q = bsxfun(@times,psijh(:,:,q),tauh(:,:,q)');
                    QL(1:r+cons,:,q) = Q(:,1:r+cons)';
                    QL(r+cons+1:end,:,q) = 0; 
                end
            end
            rs(q) = r;                 
        end
    end      

    
    
    %% =====| Step 4: Sample factors 
    for q = 1:nq
        if prior == 4 
           r = rs(q);
        end
        
        if prior == 12
            for t = 1:n
                x_tilde = (x(t,:) - k1(q).*z(t,:,q) - (cons>0).*L(1,:,q))./v(t,:,q);                                          % Standardized LHS variables
                L_tilde = L(cons+1:H(q)+cons,:,q)./v(t,:,q);                                                                     % Standardized RHS variables                     
                F(t,1:H(q),q) = randn_gibbs(x_tilde',L_tilde',H(q).*ones(H(q),1),H(q),H(q),1);       
            end
        else
            for t = 1:n
                x_tilde = (x(t,:) - k1(q).*z(t,:,q) - (cons>0).*L(1,:,q))./v(t,:,q);                                          % Standardized LHS variables
                L_tilde = L(cons+1:r+cons,:,q)./v(t,:,q);                                                                     % Standardized RHS variables                     
                F(t,1:r,q) = randn_gibbs(x_tilde',L_tilde',ones(r,1),r,r,1); 
            end
        end
    end
    
    if prior == 12
        
        for q = 1:nq
            
            % update z_star 
            lhd_spike = zeros(H(q),1);
            lhd_slab  = zeros(H(q),1);
            
            for h = 1:H(q)
                lhd_spike(h) = exp(sum(log(normpdf(L(cons+h,:,q)', 0, sqrt(theta_inf)))));
                lhd_slab(h)  = mvtpdf(L(cons+h,:,q)',(b_theta/a_theta).*eye(p),2*a_theta);   %dmvt(L(cons+h,:,q)', zeros(p, 1), (b_theta / a_theta) * eye(p), 2 * a_theta);
                prob_h = w_post(1:H(q),q) .* [repmat(lhd_spike(h), h, 1); repmat(lhd_slab(h), H(q) - h, 1)]; 
                if sum(prob_h) == 0
                    prob_h = [zeros(H(q) - 1, 1); 1];
                else
                    %prob_h = exp(log(prob_h)-max(log(prob_h)));
                    prob_h = prob_h./ sum(prob_h);
                end
                    z_star(h,q) = (1:H(q)) * (mnrnd(1, prob_h))';
            end
            % sample and update v and w
            vv = NaN(H(q),1);
            for h = 1:H(q)-1
                vv(h) = betarnd(1+sum(z_star(:,q)==h), alpha + sum(z_star(:,q)>h));%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
            vv(H(q)) = 1;
            w_post(1,q) = vv(1);
            
            for h = 2:H(q)              
               w_post(h,q) = max(vv(h) *  prod(1 - vv(1:(h-1))),0);
            end
            
            
            if sum(w_post(:,q)<0)>0
                stop
            end
            if sum(abs(w_post(:,q))>10)>0
                stop
            end
            
            % sample theta inverse
            for h = 1:H(q)
                if z_star(h,q) <= h
                    theta_inv(h,q) = 1 / theta_inf;
                else
                    theta_inv(h,q) = gamrnd(a_theta + 0.5 * p, 1 / (b_theta + 0.5 * (L(cons+h,:,q) *L(cons+h,:,q)')));
                end
            end
            
            % update H
            active   = find(z_star(1:H(q),q) > (1:H(q))');
            Hstar(q) = length(active);
            
            if irep >= start_adapt && rand <= exp(alpha0 + alpha1 *t)
                if Hstar(q) < H(q)-1
                    H(q) = Hstar(q) + 1;
                    F(:,:,q)    = [F(:,active,q), normrnd(0,1,[n,length(setdiff(1:p,active))])];
                    theta_inv(:,q) = [theta_inv(active); ones(length(setdiff(1:p,active)),1).*1/theta_inf];
                    w_post(:,q)    = max([w_post(active,q); ones(length(setdiff(1:p,active)),1).*(1-sum(w_post(:,q)))],0);
                    L(1+cons:length(active)+cons+1,:,q) = [L(1+cons:length(active)+cons,:,q); randn(1,p).*sqrt(theta_inf)];
                elseif H(q) < p
                    H(q)        = H(q) + 1;
                    F(:,H(q),q) = normrnd(0,1,[n,1]);
                    vv(H(q)-1)  = betarnd(1, alpha);
                    vv(H(q))    = 1;
                    w_post(1,1) = vv(1);
                    for h=1:H(q)
                       w_post(h,q) = max(vv(h) *  prod(1 - vv(1:(h-1))),0);
                    end
                end
            end
            
        end
        QL = repmat(reshape([ones(1,nq);theta_inv],p+1,[],nq),1,p,1);
    end

    
    if irep > nburnin && mod(irep,nthin)==0
        % Save draws of parameters
        savedraw = savedraw + 1;
        if prior == 12
            for q=1:nq
                F_draws(:,1:H(q),q,savedraw)     = F(:,1:H(q),q);
            end
            H_draws(savedraw,:) = H;
        else
            F_draws(:,:,:,savedraw)     = F;
        end
        if prior == 6 || prior == 7 || prior == 8 || prior == 9 || prior == 10 || prior == 11 
            L_draws(:,:,:,savedraw)     = L.*permute(gammaL,[2,1,3]);
        elseif prior == 12
            for q = 1:nq
                L_draws(1+cons:H(q)+cons,:,q,savedraw) = L(1+cons:H(q)+cons,:,q);
            end
        else
            L_draws(:,:,:,savedraw)     = L;
        end
        z_draws(:,:,:,savedraw)     = z;        
    end
end

F = mean(F_draws,4);
L = mean(L_draws,4);

% out.F_draws = F_draws;
% out.L_draws = L_draws;
% out.z_draws = z_draws;
% if prior == 12
%     out.H_draws = H_draws;
% end

end
