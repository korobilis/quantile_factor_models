%% MONTE_CARLO_OUTLIER.m Monte Carlo study for quantile factor regression
%==========================================================================
% We consider a factor model where the individual series are contaminated
% with various outlier processes.
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% This version: 13 February 2024
%==========================================================================

clear all;close all;clc;

addpath('functions')
addpath('models')

%% ==================| MONTE CARLO SETTIGS |========================
nMC = 500;                     % Monte Carlo iterations
% T = 100;                     % Number of observations
% p = 10;                      % Number of variables
% r = 3;                       % Number of factors
% outlier = 1;                 % Outlier process. Options: 1 Innovative, 2 Additive, 3 Level shift, 4 Ramp shift
quant = [.25,.5,.75];          % Quantile levels
nq = length(quant);
%=================================================================== 

for T = [50,100,200]
    for p = [50,100,200]
        for outlier = 1:4
            for r = 3

R2   = zeros(nMC,r,nq,3);
NUM  = zeros(nMC,nq,3);
DEN  = zeros(nMC,nq,3);
MSD  = zeros(nMC,r,nq,3);
%% Start Monte Carlo iterations
for iMC = 1:nMC  
    disp('  ')
    disp(['This is Monte Carlo iteration ' num2str(iMC) ' of ' num2str(nMC)]);
    
    %% ==========| GENERATE ARTIFICIAL DATA    
    rng('shuffle')%rng(iMC,'twister');
    %% Generate here factors and loadings to be used in all datasets
    F = zeros(T,r);
    F(1,:) = randn(1,r);
    for t = 2:T+100
        F(t,:) = 0.8*F(t-1,:) + randn(1,r);
    end
    F = F(end-(T-1):end,:);
    L = randn(p,r);
    beta = L*(0.8*eye(r))*((L'*L)\L');

    %% Generate clean data x
    x = F*L' + randn(T,p);  

    %% Generate outlier process
    h = unidrnd(round(T/2),p,1) + round(T/4);     % Outlier observations randomly between (T/4, 3*T/4)
    w = 5*rand(1,p).*abs(mean(x));                % Outlier strength randomly between (0,5*mean(x))
    % Create Indicator and Step functions
    It = zeros(T,p); St = zeros(T,p);
    for i = 1:p
        It(h(i),i) = 1;     St(h(i):end,i) = 1;
    end

    if outlier == 1 %% Case 1: Innovative outlier
        outlier_process = It(1,:);
        for t = 2:T
            outlier_process(t,:) = (It(t,:) - It(t-1,:)*beta).*w;
        end
        y = x + outlier_process;
    elseif outlier == 2 %% Case 2: Additive outlier
        outlier_process = It.*w;
        y = x + outlier_process;
    elseif outlier == 3 %% Case 3: Level shift
        outlier_process = It(2:end,:).*w;
        diffy = diff(x) + outlier_process;
        y = cumsum(diffy) + x(1,:);
    elseif outlier == 4 %% Case 4: Ramp shift
        outlier_process = St(2:end,:);
        diffy = diff(x) + outlier_process;
        y = cumsum(diffy) + x(1,:);
    end

    %% ==========| ESTIMATE QR FACTOR MODELS      
    disp(['Estimating factors for DGP ' num2str(i) ' of 10'])      
    %% Simple PCA estimates first
    FPCA = extract(zscore(y),r);
   
    %% CDG estimates next
    tic;       
    QF_CDG = zeros(T,r,nq);      
    for q = 1:nq         
        %[QF_CDG(:,:,q),~] = IQR(y,r,1e-4,quant(q));
        [QF_CDG(:,:,q),~] = IQR(zscore(y),r,1e-4,quant(q));
    end
    disp('QPCA finished'); toc;

    %% VBQFA estimates next   
    tic;
    [QF_VB] = VBQFA(y,r,quant,0,300,2);   
    disp('VBQFA finished'); toc;
    
    %% MCMC estimates next
    tic;
    [QF_MCMC] = MCMCQFA(y,r,quant,0,1000,1000,10,1,2);
    disp('MCMC finished'); toc;
          
    %% save results
    Pf = F*inv(F'*F)*F'; Fz = zscore(F);   
    for q = 1:nq
        r2CDG = []; r2VB = []; r2MCMC = [];
        for j = 1:r
            r2CDG  = [r2CDG, rsquare_a(F(:,j),QF_CDG(:,:,q))]; %#ok<*AGROW> 
            r2VB   = [r2VB, rsquare_a(F(:,j),QF_VB(:,:,q))];
            r2MCMC = [r2MCMC, rsquare_a(F(:,j),QF_MCMC(:,:,q))];
        end

        % CDG2021 R2 statistic           
        R2(iMC,:,q,1) = r2CDG;
        R2(iMC,:,q,2) = r2VB;
        R2(iMC,:,q,3) = r2MCMC;   
          
        % SW2002 trace statistic           
        NUM(iMC,q,1) = trace(squeeze(QF_CDG(:,:,q))'*Pf*squeeze(QF_CDG(:,:,q)));       
        NUM(iMC,q,2) = trace(squeeze(QF_VB(:,:,q))'*Pf*squeeze(QF_VB(:,:,q)));
        NUM(iMC,q,3) = trace(squeeze(QF_MCMC(:,:,q))'*Pf*squeeze(QF_MCMC(:,:,q)));

        DEN(iMC,q,1) = trace(squeeze(QF_CDG(:,:,q))'*squeeze(QF_CDG(:,:,q)));
        DEN(iMC,q,2) = trace(squeeze(QF_VB(:,:,q))'*squeeze(QF_VB(:,:,q)));
        DEN(iMC,q,3) = trace(squeeze(QF_MCMC(:,:,q))'*squeeze(QF_MCMC(:,:,q)));

        % MSD statistic
        MSD(iMC,:,q,1) =  mean((Fz - zscore(QF_CDG(:,:,q))).^2);          
        MSD(iMC,:,q,2) =  mean((Fz - zscore(QF_VB(:,:,q))).^2);
        MSD(iMC,:,q,3) =  mean((Fz - zscore(QF_MCMC(:,:,q))).^2);     
    end
end
% Compute trace statistic as the expectation of the numerator and
% denominator terms 
TR = zeros(nq,3);
for q = 1:nq
    TR(q,1) = mean(NUM(:,q,1))./mean(DEN(:,q,1));   
    TR(q,2) = mean(NUM(:,q,2))./mean(DEN(:,q,2));
    TR(q,3) = mean(NUM(:,q,3))./mean(DEN(:,q,3));
end
save(sprintf('%s_%g_%g_%g_%g.mat','MONTE_CARLO_OUTLIER',T,p,r,outlier),'-mat');
            end
        end
    end
end


