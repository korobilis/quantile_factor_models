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
% A = readtable('FRED_WD.xlsx','VariableNamingRule','preserve','Sheet','From 2008');
A = readtable('NFCI.xlsx','Sheet','NFCI');
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
for sample = T_full:T_full
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
    F   = ([FPCA,NFCI(1:sample,:),QFCDG,QFVB]);
    F_z = zscore([FPCA,NFCI(1:sample,:),QFCDG,QFVB]);

    
end %recursive forecasts


%% In-sample figure

FigH = figure('Position', get(0, 'Screensize'));    
hold on 
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,2),'-','Color',[0 0.4470 0.7410],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
title("NFCI Index")
ax = gca();
ax.FontSize = 20;
saveas(FigH, ["NFCI_NFCI.jpg"],'jpeg');

FigH = figure('Position', get(0, 'Screensize'));    
hold on 
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,1),'-','Color',[0 0.4470 0.7410],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
title("PCA factor")
ax = gca();
ax.FontSize = 20;
saveas(FigH, ["NFCI_PCA.jpg"],'jpeg');

FigH = figure('Position', get(0, 'Screensize'));    
hold on 
%plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,2),'-.','Color','black','Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,end-2),'Color',[0 0.4470 0.7410],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F(:,end-1),'--','Color',[0.8500 0.3250 0.0980],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,end),':','Color',[0.9290 0.6940 0.1250],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
title("Probabilistic quantile factors")
ax = gca();
ax.FontSize = 20;
saveas(FigH, ["NFCI_VBQFA.jpg"],'jpeg');

FigH = figure('Position', get(0, 'Screensize'));    
hold on 
%plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,2),'-.','Color','black','Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,3),'Color',[0 0.4470 0.7410],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F(:,4),'--','Color',[0.8500 0.3250 0.0980],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,5),':','Color',[0.9290 0.6940 0.1250],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
title("Nonparametric quantile factors")
ax = gca();
ax.FontSize = 20;
saveas(FigH, ["NFCI_CDG.jpg"],'jpeg');

%% 


FigH = figure('Position', get(0, 'Screensize'));  
subplot(2,2,1)
hold on 
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,2),'-','Color',[0 0.4470 0.7410],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
box on
title("NFCI Index")
ax = gca();
ax.FontSize = 15;

subplot(2,2,2)
hold on 
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,1),'-','Color',[0 0.4470 0.7410],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
box on
title("PCA factor")
ax = gca();
ax.FontSize = 15;

subplot(2,2,3)
hold on 
%plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,2),'-.','Color','black','Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,end-2),'Color',[0 0.4470 0.7410],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F(:,end-1),'--','Color',[0.8500 0.3250 0.0980],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,end),':','Color',[0.9290 0.6940 0.1250],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
box on
title("Probabilistic quantile factors")
ax = gca();
ax.FontSize = 15;

subplot(2,2,4)
hold on 
%plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,2),'-.','Color','black','Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,3),'Color',[0 0.4470 0.7410],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),-1*F(:,4),'--','Color',[0.8500 0.3250 0.0980],'Linewidth',2)
plot(datetime(table2array(A(1:end-nfore-1,1))),F_z(:,5),':','Color',[0.9290 0.6940 0.1250],'Linewidth',2)
%plot(datetime(table2array(A(1:end-nfore-1,1))),mean(F(:,1),2),'Linewidth',2)
hold off
grid on
box on
title("Nonparametric quantile factors")
ax = gca();
ax.FontSize = 15;

saveas(FigH, ["NFCI_factors.eps"],'epsc2');