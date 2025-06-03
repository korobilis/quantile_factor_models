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
nMC = 1;                   % Monte Carlo iterations
T = 100;                     % Number of observations
p = 50;                      % Number of variables
r = 6;                       % Number of factors
quant = [.25,.5,.75];        % Quantile levels
nq = length(quant);
%=================================================================== 
R2   = zeros(nMC,r,nq,2);
NUM  = zeros(nMC,nq,2);
DEN  = zeros(nMC,nq,2);
MSD  = zeros(nMC,r,nq,2);
%% Start Monte Carlo iterations
for iMC = 1:nMC  
    disp('  ')
    disp(['This is Monte Carlo iteration ' num2str(iMC) ' of ' num2str(nMC)]);
    
    %% ==========| GENERATE ARTIFICIAL DATA    
    rng('shuffle')%rng(iMC,'twister');
    %% Generate here factors and loadings to be used in all datasets
    F = zeros(T,r);
    F(1,:) = rand(1,r);
    for t = 2:T+100
        F(t,:) = 0.8*F(t-1,:) + randn(1,r);
    end
    F = F(end-(T-1):end,:);
    L = randn(p,r);  

    %% Generate 8 different error distributions
    error  = trnd(3,T,p);                                       % CDD2021 DGP1
    %% Generate data y
    y = F*L' + error;

    %% ==========| ESTIMATE QR FACTOR MODELS     
    %% Simple PCA estimates first   
    FPCA = extract(zscore(y),r);
       
    %% full VBFA estimates next
    tic;      
    [QF_VB1,L_VB1,ELBOsave1,kappasave1,mu_fsave1] = VBQFA(y,1,300,quant,0,2);
    disp('VBQFA finished'); toc;

    tic;      
    [QF_VB2,L_VB2,ELBOsave2,kappasave2,mu_fsave2] = VBQFA(y,2,300,quant,0,2);
    disp('VBQFA finished'); toc;

    tic;      
    [QF_VB3,L_VB3,ELBOsave3,kappasave3,mu_fsave3] = VBQFA(y,3,300,quant,0,2);
    disp('VBQFA finished'); toc;

    tic;      
    [QF_VB4,L_VB4,ELBOsave4,kappasave4,mu_fsave4] = VBQFA(y,4,300,quant,0,2);
    disp('VBQFA finished'); toc;

    tic;      
    [QF_VB5,L_VB5,ELBOsave5,kappasave5,mu_fsave5] = VBQFA(y,5,300,quant,0,2);
    disp('VBQFA finished'); toc;

    tic;      
    [QF_VB6,L_VB6,ELBOsave6,kappasave6,mu_fsave6] = VBQFA(y,6,300,quant,0,2);
    disp('VBQFA finished'); toc;

   [~,elbo(iMC,:)] = max([ELBOsave1(end) ELBOsave2(end) ELBOsave3(end) ELBOsave4(end) ELBOsave5(end) ELBOsave6(end)]);
    
end

%save(sprintf('%s_%g_%g_%g.mat','MONTE_CARLO',T,p,r),'-mat');


