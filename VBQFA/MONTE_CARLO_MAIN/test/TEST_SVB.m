%% MONTE_CARLO_CDG.m Monte Carlo study for quantile factor regression
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

for T = [500]
    for p = [50]
        for r = 3

R2   = zeros(nMC,r,nq,8,2);
NUM  = zeros(nMC,nq,8,2);
DEN  = zeros(nMC,nq,8,2);
MSD  = zeros(nMC,r,nq,8,2);
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
    error(:,:,8) = (probs<=9/20).*Normal(-6/5,(3/5)^2,T,p) + (probs>9/20 & probs<=18/20).*Normal(6/5,(3/5)^2,T,p) + (probs>18/20).*Normal(0,(1/4)^2,T,p);
    %% Generate data y
    y = zeros(T,p,8);
    for i = 1:8
        y(:,:,i) = F*L' + error(:,:,i);
    end

    %% ==========| ESTIMATE QR FACTOR MODELS
    for i = 1:8      
        disp(['Estimating factors for DGP ' num2str(i) ' of 10'])
        %% Simple PCA estimates first
        [FPCA,Lpca] = extract(zscore(y(:,:,i)),r);
        [VB1,LVB1] = VBFA(y(:,:,i),r,300,0,2);
        [VB2,LVB2]  = SVBFA(y(:,:,i),r,300,0,2,1);

        [mean(mean((y(:,:,i)-FPCA*Lpca').^2)) mean(mean((y(:,:,i)-VB1(:,:,1)*LVB1(:,:,1)).^2)) mean(mean((y(:,:,i)-VB2(:,:,1)*LVB2(:,:,1)).^2))]
        
        ll = Lpca;
        fhat = FPCA;
        sigmaF=fhat'*fhat/T;
        sigmaA=ll'*ll/p;
        
        dum1= (sigmaF)^(0.5)*sigmaA*(sigmaF)^(0.5);
        [dum2,dum3,dum4]=svd(dum1);
        R= (sigmaA)^(-0.5)*dum2;
        fhat = fhat* (inv(R))';
        ll=ll* R;
        
        ff = y(:,:,i)*ll/p;
        fhat_PCA = ff*((ff'*ff)/T)^(0.5);
                
        
        ll = LVB1(:,:,1)';
        fhat = VB1(:,:,1);
        sigmaF=fhat'*fhat/T;
        sigmaA=ll'*ll/p;
        
        dum1= (sigmaF)^(0.5)*sigmaA*(sigmaF)^(0.5);
        [dum2,dum3,dum4]=svd(dum1);
        R= (sigmaA)^(-0.5)*dum2;
        fhat = fhat* (inv(R))';
        ll=ll* R;
        
        ff = y(:,:,i)*ll/p;
        fhat_VB1 = ff*((ff'*ff)/T)^(0.5);
               
       
        ll = LVB2(:,:,1)';
        fhat = VB2(:,:,1);
        sigmaF=fhat'*fhat/T;
        sigmaA=ll'*ll/p;
        
        dum1= (sigmaF)^(0.5)*sigmaA*(sigmaF)^(0.5);
        [dum2,dum3,dum4]=svd(dum1);
        R= (sigmaA)^(-0.5)*dum2;
        fhat = fhat* (inv(R))';
        ll=ll* R;
        
        ff = y(:,:,i)*ll/p;
        fhat_VB2= ff*((ff'*ff)/T)^(0.5);
        
        %% CDG estimates next
        tic;
%         QF_CDG = zeros(T,r,nq);
%         for q = 1:nq
%             %[QF_CDG(:,:,q),~] = IQR(y(:,:,i),r,1e-4,quant(q));
%             [QF_CDG(:,:,q),~] = IQR(zscore(y(:,:,i)),r,1e-4,quant(q));
%         end
%         disp('QPCA finished'); toc;

        %% full VBFA estimates next
        tic;
        [QF_VB]  = VBQFA(y(:,:,i),r,300,quant,0,2);
        toc;
        tic;
        [QF_VB2] = SVBQFA(y(:,:,i),r,300,quant,0,2,[]);
        toc;
        disp('VBQFA finished'); toc;
        
        %% save results
        Pf = F*inv(F'*F)*F'; Fz = zscore(F);
        for q = 1:nq
            r2CDG = []; r2VB = [];
            for j = 1:r
                r2CDG  = [r2CDG, rsquare_a(F(:,j),QF_CDG(:,:,q))]; %#ok<*AGROW> 
                r2VB   = [r2VB, rsquare_a(F(:,j),QF_VB(:,:,q))];
            end

            % CDG2021 R2 statistic
            R2(iMC,:,q,i,1) = r2CDG;
            R2(iMC,:,q,i,2) = r2VB;            

            % SW2002 trace statistic           
            NUM(iMC,q,i,1) = trace(squeeze(QF_CDG(:,:,q))'*Pf*squeeze(QF_CDG(:,:,q)));
            NUM(iMC,q,i,2) = trace(squeeze(QF_VB(:,:,q))'*Pf*squeeze(QF_VB(:,:,q)));

            DEN(iMC,q,i,1) = trace(squeeze(QF_CDG(:,:,q))'*squeeze(QF_CDG(:,:,q)));
            DEN(iMC,q,i,2) = trace(squeeze(QF_VB(:,:,q))'*squeeze(QF_VB(:,:,q)));

            % MSD statistic
            MSD(iMC,:,q,i,1) =  mean((Fz - zscore(QF_CDG(:,:,q))).^2);
            MSD(iMC,:,q,i,2) =  mean((Fz - zscore(QF_VB(:,:,q))).^2);           
        end
    end
end
% Compute trace statistic as the expectation of the numerator and
% denominator terms 
TR = zeros(nq,10,2);
for i = 1:8
    for q = 1:nq
        TR(q,i,1) = mean(NUM(:,q,i,1))./mean(DEN(:,q,i,1));
        TR(q,i,2) = mean(NUM(:,q,i,2))./mean(DEN(:,q,i,2));
    end
end
save(sprintf('%s_%g_%g_%g_20230516.mat','MONTE_CARLO',T,p,r),'-mat');
        end
    end
end
