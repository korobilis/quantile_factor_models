% Reset everything, clear memory and screen
clear; 
close all; 
clc;

% Start clock
tic;
% Add path of random number generators
addpath('functions')
addpath('models')
addpath('data')
 
%===========================| USER INPUT |================================= 
% Quantile factor specification
quants = [0.1,0.5,0.9];
nq = length(quants);

% VAR specification
p      = 12;             % Number of lags
nfore  = 24;             % Number of predictions steps ahead

%% ===========================| LOAD DATA |=================================
% Read in the data
[A,B,~] = xlsread('alldata.xlsx','Indices');
EPU = A(:,1);
Xraw = A(:,2:end);
Yraw = xlsread('alldata.xlsx','Macro');
%---------------

% Start with  first_per% of observations for first period estimation
first_per = 0.4;
T_full    = (size(Yraw,1)-2-nfore);
T_thres   = round(first_per*T_full); 
anumber   = T_full-T_thres+1;

MSFE  = zeros(anumber,nfore,size(Yraw,2),8);    % Store MSFEs here

for sample = T_thres:T_full
    disp('     ')
    disp(['Now you are running sample ' num2str(sample) ' of ' num2str(T_full)] )      
       
    % in-sample data
    Y = Yraw(1:sample,:);
    X = Xraw(1:sample,:);
    FEPU = EPU(1:sample,:);

    % Out-of-sample observations
    Y_fore = Yraw(sample+1:sample+nfore,:);
     
    %% FACTOR ESTIMATION
    % |==== PCA factors
    [FPCA,~] = extract(zscore(X),1);
    % |==== CDG quantile factors
    QFCDG = zeros(sample,nq);
    for q = 1:nq   
        [QFCDG(:,q), ~] = IQR(zscore(X),1,1e-4,quants(q));
        corrCDG = corrcoef(QFCDG(:,q),FEPU);
        QFCDG(:,q) = sign(corrCDG(1,2))*QFCDG(:,q);
    end    
    % |==== VB quantile factors with shrinkage
    [QFVB,~] = VBQFA(zscore(X), 1, 300, quants, 0, 2);
    QFVB = squeeze(QFVB);
    for q = 1:nq   
        corrVB = corrcoef(QFVB(:,q),FEPU);
        QFVB(:,q) = sign(corrVB(1,2))*QFVB(:,q);
    end 

    % Collect all factors used in forecast comparison
    F = [FPCA,FEPU,QFCDG,QFVB];

    %% VAR ESTIMATION
    Forecast = zeros(nfore,size(Y,2)+1,size(F,2));    
    for imodel = 1:size(F,2)
        YF = [Y,F(:,imodel)];
        Mdl = varm(size(YF,2),p);
        EstMdl = estimate(Mdl,YF);
        [Forecast(:,:,imodel),~] = forecast(EstMdl,nfore,YF);        
        MSFE(sample-T_thres+1,:,:,imodel)   = (Forecast(:,1:size(Y,2),imodel) - Y_fore).^2;  
    end
end %recursive forecasts


%% Print tables
%1) IP
PCA=squeeze(mean(MSFE(:,[1:6,12,24],1,1)))'./squeeze(mean(MSFE(:,[1:6,12,24],1,2)))';
QFCDG10=squeeze(mean(MSFE(:,[1:6,12,24],1,3)))'./squeeze(mean(MSFE(:,[1:6,12,24],1,2)))';
QFCDG50=squeeze(mean(MSFE(:,[1:6,12,24],1,4)))'./squeeze(mean(MSFE(:,[1:6,12,24],1,2)))';
QFCDG90=squeeze(mean(MSFE(:,[1:6,12,24],1,5)))'./squeeze(mean(MSFE(:,[1:6,12,24],1,2)))';
QFVB10=squeeze(mean(MSFE(:,[1:6,12,24],1,6)))'./squeeze(mean(MSFE(:,[1:6,12,24],1,2)))';
QFVB50=squeeze(mean(MSFE(:,[1:6,12,24],1,7)))'./squeeze(mean(MSFE(:,[1:6,12,24],1,2)))';
QFVB90=squeeze(mean(MSFE(:,[1:6,12,24],1,8)))'./squeeze(mean(MSFE(:,[1:6,12,24],1,2)))';
T1 = table(PCA,QFCDG10,QFCDG50,QFCDG90,QFVB10,QFVB50,QFVB90,'RowNames',{'h=1';'h=2';'h=3';'h=4';'h=5';'h=6';'h=12';'h=24'});

%2) CPI
PCA=squeeze(mean(MSFE(:,[1:6,12,24],2,1)))'./squeeze(mean(MSFE(:,[1:6,12,24],2,2)))';
QFCDG10=squeeze(mean(MSFE(:,[1:6,12,24],2,3)))'./squeeze(mean(MSFE(:,[1:6,12,24],2,2)))';
QFCDG50=squeeze(mean(MSFE(:,[1:6,12,24],2,4)))'./squeeze(mean(MSFE(:,[1:6,12,24],2,2)))';
QFCDG90=squeeze(mean(MSFE(:,[1:6,12,24],2,5)))'./squeeze(mean(MSFE(:,[1:6,12,24],2,2)))';
QFVB10=squeeze(mean(MSFE(:,[1:6,12,24],2,6)))'./squeeze(mean(MSFE(:,[1:6,12,24],2,2)))';
QFVB50=squeeze(mean(MSFE(:,[1:6,12,24],2,7)))'./squeeze(mean(MSFE(:,[1:6,12,24],2,2)))';
QFVB90=squeeze(mean(MSFE(:,[1:6,12,24],2,8)))'./squeeze(mean(MSFE(:,[1:6,12,24],2,2)))';
T2 = table(PCA,QFCDG10,QFCDG50,QFCDG90,QFVB10,QFVB50,QFVB90,'RowNames',{'h=1';'h=2';'h=3';'h=4';'h=5';'h=6';'h=12';'h=24'});

%3) FFR
PCA=squeeze(mean(MSFE(:,[1:6,12,24],3,1)))'./squeeze(mean(MSFE(:,[1:6,12,24],3,2)))';
QFCDG10=squeeze(mean(MSFE(:,[1:6,12,24],3,3)))'./squeeze(mean(MSFE(:,[1:6,12,24],3,2)))';
QFCDG50=squeeze(mean(MSFE(:,[1:6,12,24],3,4)))'./squeeze(mean(MSFE(:,[1:6,12,24],3,2)))';
QFCDG90=squeeze(mean(MSFE(:,[1:6,12,24],3,5)))'./squeeze(mean(MSFE(:,[1:6,12,24],3,2)))';
QFVB10=squeeze(mean(MSFE(:,[1:6,12,24],3,6)))'./squeeze(mean(MSFE(:,[1:6,12,24],3,2)))';
QFVB50=squeeze(mean(MSFE(:,[1:6,12,24],3,7)))'./squeeze(mean(MSFE(:,[1:6,12,24],3,2)))';
QFVB90=squeeze(mean(MSFE(:,[1:6,12,24],3,8)))'./squeeze(mean(MSFE(:,[1:6,12,24],3,2)))';
T3 = table(PCA,QFCDG10,QFCDG50,QFCDG90,QFVB10,QFVB50,QFVB90,'RowNames',{'h=1';'h=2';'h=3';'h=4';'h=5';'h=6';'h=12';'h=24'});


% display table
clc;
disp('MSFEs of IP')
disp(T1);
disp('  ' )
disp('MSFEs of CPI')
disp(T2);
disp('  ' )
disp('MSFEs of FFR')
disp(T3);

save(sprintf('%s_%g_%g.mat','VAR_FORECASTING',p,nfore),'-mat');
