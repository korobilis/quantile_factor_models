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
quants = [0.10,0.5,0.90];
nq = length(quants);

% VAR specification
p      = 2;              % Number of lags
nfore  = 26;             % Number of predictions steps ahead

%% ===========================| LOAD DATA |=================================
A = readtable('NFCI.xlsx','VariableNamingRule','preserve','Sheet','NFCI');
tcode     = table2array(A(1,2:end));
data   = table2array(A(2:end,2:end));

% Assign X and Y variables
Xraw = data(:,2:end-1);
NFCI = data(:,end);
Yraw = data(:,1);
%---------------

% Start with  first_per% of observations for first period estimation
first_per = 0.5;
T_full    = (size(Yraw,1)-nfore);
T_thres   = round(first_per*T_full); 
anumber   = T_full-T_thres+1;

MSFE  = zeros(anumber,nfore,size(Yraw,2),7);    % Store MSFEs here
for sample = T_thres:T_full
    disp('     ')
    disp(['Now you are running sample ' num2str(sample) ' of ' num2str(T_full)] )      
       
    Y = Yraw(1:sample,:);
    X = Xraw(1:sample,:);
    % Out-of-sample observations
    Y_fore = Yraw(sample+1:sample+nfore,:);
     
    %% FACTOR ESTIMATION
    % |==== PCA factors
    [FPCA,~] = extract(normalize(X),1);
    C = corrcoef([FPCA,NFCI(1:sample,:)]);
    FPCA = sign(C(1,2))*FPCA;
    % |==== CDG quantile factors
    QFCDG = repmat(0*FPCA,1,nq);
    for q = 1:nq         
        [QFCDG(:,q), ~] = IQR(normalize(X),1,1e-5,quants(q));
        corrCDG = corrcoef(QFCDG(:,q),FPCA);
        QFCDG(:,q) = sign(corrCDG(1,2))*QFCDG(:,q);
    end    
    % |==== VB quantile factors with shrinkage
    [QFVB,~] = VBQFA(normalize(X), 1, 100, quants, 0, 2);
    QFVB = squeeze(QFVB);
    for q = 1:nq
        corrVB = corrcoef(QFVB(:,q),FPCA);   
        QFVB(:,q) = sign(corrVB(1,2))*QFVB(:,q);
    end     

    % Collect all factors used in forecast comparison
    F = zscore([FPCA,NFCI(1:sample,:),QFCDG,QFVB]);

    %% VAR ESTIMATION
    Forecast = zeros(nfore,size(Y,2)+1,size(F,2));    
    for imodel = 1:size(F,2)
        YF = [Y,F(:,imodel)];
        Mdl = varm(size(YF,2),p);
        EstMdl = estimate(Mdl,YF);
        [Forecast(:,:,imodel),~] = forecast(EstMdl,nfore,YF);        
        MSFE(sample-T_thres+1,:,:,imodel)   = (Forecast(:,1:size(Y,2),imodel) - Y_fore).^2;  
    end
           
    YF = Y;   
    Mdl = varm(size(YF,2),p);
    EstMdl = estimate(Mdl,YF);
    [Forecast(:,1,imodel+1),~] = forecast(EstMdl,nfore,YF);        
    MSFE(sample-T_thres+1,:,:,imodel+1)   = (Forecast(:,1:size(Y,2),imodel+1) - Y_fore).^2;  
end %recursive forecasts


%% Print tables
%1) WEI
PCA=squeeze(mean(MSFE(:,1:nfore,1,1)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
NFCI=squeeze(mean(MSFE(:,1:nfore,1,2)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
QFCDG10=squeeze(mean(MSFE(:,1:nfore,1,3)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
QFCDG50=squeeze(mean(MSFE(:,1:nfore,1,4)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
QFCDG90=squeeze(mean(MSFE(:,1:nfore,1,5)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
QFVB10=squeeze(mean(MSFE(:,1:nfore,1,6)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
QFVB50=squeeze(mean(MSFE(:,1:nfore,1,7)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
QFVB90=squeeze(mean(MSFE(:,1:nfore,1,8)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
ARp=squeeze(mean(MSFE(:,1:nfore,1,9)))'./squeeze(mean(MSFE(:,1:nfore,1,2)))';
T1 = table(PCA,NFCI,QFCDG10,QFCDG50,QFCDG90,QFVB10,QFVB50,QFVB90,ARp,'RowNames',{'h=1';'h=2';'h=3';'h=4';'h=5';'h=6';'h=7';'h=8';'h=9';'h=10';'h=11';'h=12';'h=13';...
                                                                            'h=14';'h=15';'h=16';'h=17';'h=18';'h=19';'h=20';'h=21';'h=22';'h=23';'h=24';'h=25';'h=26'});

% display table
clc;
disp('MSFEs of WEI')
disp(T1);


save(sprintf('%s_%g_%g.mat','NFCI_FORECASTING',p,nfore),'-mat');

