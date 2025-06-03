% Reset everything, clear memory and screen
clear; close all; clc;
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

%% ===========================| LOAD DATA |=================================
% Read in the data
[A,B,~] = xlsread('alldata.xlsx','Indices');
EPU = A(:,1);
Xraw = A(:,2:end);
Yraw = xlsread('alldata.xlsx','Macro');
dates = B(2:end,:);

Y = Yraw(1:end,:);
X = Xraw(1:end,:);

%% FACTOR ESTIMATION
% |==== PCA factors
[FPCA,~] = extract(normalize(X),1);
C = corrcoef([FPCA,EPU(1:end,:)]);
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

% Plot insample factor estimates
graph = panel();
graph.fontname = 'Arial';
graph.fontsize = 16;
graph.pack(2, 2);
graph(1,1).select();
plot(zscore(EPU),'Linewidth',3);
grid on; box on;
xlim([1 size(Y,1)])
set(gca,'XTick',1:48:size(Y,1))
set(gca,'XTicklabels',dates(1:48:size(Y,1)))
title('Total EPU Index')
graph(1, 2).select();
plot(zscore(FPCA),'Linewidth',3);
grid on; box on;
xlim([1 size(Y,1)])
set(gca,'XTick',1:48:size(Y,1))
set(gca,'XTicklabels',dates(1:48:size(Y,1)))
title('PCA factor')
graph(2,1).select();
plot((1:size(Y,1))',QFVB(:,1),'-',(1:size(Y,1))',QFVB(:,2),'--',(1:size(Y,1))',QFVB(:,3),':','Linewidth',3);
legend({'10% factor';'50% factor';'90% factor'},'Location','northwest')
grid on; box on;
xlim([1 size(Y,1)])
set(gca,'XTick',1:48:size(Y,1))
set(gca,'XTicklabels',dates(1:48:size(Y,1)))
title('Probabilistic quantile factors')
graph(2,2).select();
plot((1:size(Y,1))',QFCDG(:,1),'-',(1:size(Y,1))',QFCDG(:,2),'--',(1:size(Y,1))',QFCDG(:,3),':','Linewidth',3);
legend({'10% factor';'50% factor';'90% factor'},'Location','northwest')
grid on; box on;
xlim([1 size(Y,1)])
set(gca,'XTick',1:48:size(Y,1))
set(gca,'XTicklabels',dates(1:48:size(Y,1)))
title('Nonparametric quantile factors')
graph.margin = [40 30 40 20];