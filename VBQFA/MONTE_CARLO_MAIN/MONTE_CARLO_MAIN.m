%% MONTE_CARLO_MAIN.m Monte Carlo study for quantile factor regression
%==========================================================================
%
% We consider the following 8 error distributions:
%     1) Student-t with 3 dof
%     2) Skewed  : 1/5 N(-22/25,1^2) + 1/5 N(-49/125,(3/2)^2) + 3/5 N(49/250,(5/9)^2)
%     3) Kurtotic: 2/3 N(0,1^2) + 1/3 N(0,(1/10)^2)
%     4) Outlier : 1/10 N(0,1^2) + 9/10 N(0,(1/10)^2)
%     5) Bimodal : 1/2 N(-1,(2/3)^2) + 1/2 N(1,(2/3)^2)
%     6) Bimodal, separate modes: 1/2 N(-3/2,(1/2)^2) + 1/2 N(3/2,(1/2)^2)
%     7) Skewed bimodal: 3/4 N(-43/100,1^2) + 1/4 N(107/100,(1/3)^2)
%     8) Trimodal: 9/20 N(-6/5,(3/5)^2) + 9/20 N(6/5,(3/5)^2) + 1/10 N(0,(1/4)^2)
%==========================================================================
% Written by Dimitris Korobilis, University of Glasgow
% This version: 22 November 2022
%==========================================================================

clear all;close all;clc;

addpath('functions')
addpath('models')

%% ==================| MONTE CARLO SETTIGS |========================
nMC = 500;                   % Monte Carlo iterations
% T = 100;                     % Number of observations
% p = 10;                      % Number of variables
% r = 3;                       % Number of factors
quant = [.25,.5,.75];        % Quantile levels
nq = length(quant);
%=================================================================== 

for T = [50,100,200]
    for p = [50,100,200]
        for r = 3

R2   = zeros(nMC,r,nq,8,3);
NUM  = zeros(nMC,nq,8,3);
DEN  = zeros(nMC,nq,8,3);
MSD  = zeros(nMC,r,nq,8,3);
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

    error          = zeros(T,p,8);    probs = rand(T,1);
    %% Generate 8 different error distributions
    error(:,:,1)  = trnd(3,T,p);                                       % CDD2021 DGP1
    error(:,:,2)  = (probs<=1/5).*Normal(-22/25,1^2,T,p) + (probs<=2/5 & probs>1/5).*Normal(-49/125,(3/2)^2,T,p) + (probs>2/5).*Normal(49/250,(5/9)^2,T,p);
    error(:,:,3)  = (probs>1/3).*Normal(0,1^2,T,p) + (probs<=1/3).*Normal(0,(1/10)^2,T,p);
    error(:,:,4)  = (probs<=1/10).*Normal(0,1^2,T,p) + (probs>1/10).*Normal(0,(1/10)^2,T,p);
    error(:,:,5)  = (probs<=1/2).*Normal(-1,(2/3)^2,T,p) + (probs>1/2).*Normal(1,(2/3)^2,T,p);
    error(:,:,6)  = (probs<=1/2).*Normal(-3/2,(1/2)^2,T,p) + (probs>1/2).*Normal(3/2,(1/2)^2,T,p);
    error(:,:,7)  = (probs>=3/4).*Normal(-43/100,1^2,T,p) + (probs<3/4).*Normal(107/100,(1/3)^2,T,p);
    error(:,:,8)  = (probs<=9/20).*Normal(-6/5,(3/5)^2,T,p) + (probs>9/20 & probs<=18/20).*Normal(6/5,(3/5)^2,T,p) + (probs>18/20).*Normal(0,(1/4)^2,T,p);
    %% Generate data y
    y = zeros(T,p,8);
    for i = 1:8
        y(:,:,i) = F*L' + error(:,:,i);
    end

    %% ==========| ESTIMATE QR FACTOR MODELS
    for i = 1:8      
        disp(['Estimating factors for DGP ' num2str(i) ' of 10'])
        %% Simple PCA estimates first
        FPCA = extract(zscore(y(:,:,i)),r);

        %% CDG estimates next
        tic;
        QF_CDG = zeros(T,r,nq);
        for q = 1:nq
            %[QF_CDG(:,:,q),~] = IQR(y(:,:,i),r,1e-4,quant(q));
            [QF_CDG(:,:,q),~] = IQR(zscore(y(:,:,i)),r,1e-4,quant(q));
        end
        disp('QPCA finished'); toc;

        %% VBQFA estimates next
        tic;
        [QF_VB] = VBQFA(y(:,:,i),r,300,quant,0,2);
        disp('VBQFA finished'); toc;
        
        %% MCMC estimates next
        tic;
        [QF_MCMC] = MCMCQFA(y(:,:,i),r,quant,0,1000,1000,10,1,2);
        disp('VBQFA finished'); toc;        
        
        %% save results
        Pf = F*inv(F'*F)*F'; Fz = zscore(F);
        for q = 1:nq
            r2CDG = []; r2VB = [];
            for j = 1:r
                r2CDG  = [r2CDG, rsquare_a(F(:,j),QF_CDG(:,:,q))]; %#ok<*AGROW> 
                r2VB   = [r2VB, rsquare_a(F(:,j),QF_VB(:,:,q))];
                r2MCMC   = [r2MCMC, rsquare_a(F(:,j),QF_MCMC(:,:,q))];
            end

            % CDG2021 R2 statistic
            R2(iMC,:,q,i,1) = r2CDG;
            R2(iMC,:,q,i,2) = r2VB;
            R2(iMC,:,q,i,2) = r2VB;

            % SW2002 trace statistic           
            NUM(iMC,q,i,1) = trace(squeeze(QF_CDG(:,:,q))'*Pf*squeeze(QF_CDG(:,:,q)));
            NUM(iMC,q,i,2) = trace(squeeze(QF_VB(:,:,q))'*Pf*squeeze(QF_VB(:,:,q)));
            NUM(iMC,q,i,3) = trace(squeeze(QF_MCMC(:,:,q))'*Pf*squeeze(QF_MCMC(:,:,q)));

            DEN(iMC,q,i,1) = trace(squeeze(QF_CDG(:,:,q))'*squeeze(QF_CDG(:,:,q)));
            DEN(iMC,q,i,2) = trace(squeeze(QF_VB(:,:,q))'*squeeze(QF_VB(:,:,q)));
            DEN(iMC,q,i,3) = trace(squeeze(QF_MCMC(:,:,q))'*squeeze(QF_MCMC(:,:,q)));

            % MSD statistic
            MSD(iMC,:,q,i,1) =  mean((Fz - zscore(QF_CDG(:,:,q))).^2);
            MSD(iMC,:,q,i,2) =  mean((Fz - zscore(QF_VB(:,:,q))).^2);
            MSD(iMC,:,q,i,3) =  mean((Fz - zscore(QF_MCMC(:,:,q))).^2);
        end
    end
end
% Compute trace statistic as the expectation of the numerator and
% denominator terms 
TR = zeros(nq,10,3);
for i = 1:8
    for q = 1:nq
        TR(q,i,1) = mean(NUM(:,q,i,1))./mean(DEN(:,q,i,1));
        TR(q,i,2) = mean(NUM(:,q,i,2))./mean(DEN(:,q,i,2));
        TR(q,i,3) = mean(NUM(:,q,i,3))./mean(DEN(:,q,i,3));
    end
end
save(sprintf('%s_%g_%g_%g_20230516.mat','MONTE_CARLO',T,p,r),'-mat');
        end
    end
end


