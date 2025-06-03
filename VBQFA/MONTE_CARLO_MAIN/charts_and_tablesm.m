clear;
clc;
close all;

addpath('functions');

%% |=== Script to Compile Tables and Charts

files = ["50_50_3",...
         "50_100_3",...
         "50_200_3",...
         "100_50_3",...
         "100_100_3",...
         "100_200_3",...
         "200_50_3",...
         "200_100_3",...
         "200_200_3"];

model_names = ["CDG",...
               "VBQFA"];
           
% Set Dimenions
m  = 10;                    % number of DGPs
s  = length(files);         % number of MC specifications 
nq = 3;                     % number of quantile levels
nm = length(model_names);   % number of models 


%% |==== Trace R2 Only
%  -------------------

% % For Each model:
%                            Trace R2
%          (50,50)  (50,100) ....        (200,200) 
%          ---------------------------------------
%    (q1) |
% M1 (q2) |
%    (q3) |
%    (q1) |
% M2 (q2) |
%    (q3) |
%    ....


% |==== Load file
TR2 = [];
TR3 = [];
for file = 1:length(files)
    load(strcat("MONTE_CARLO_",files(file),".mat"))
    
    % data for tables
    TR2 = cat(2,TR2,reshape(real(TR),[],1,nm));
    
    % data for charts
    TR3 = cat(4,TR3,permute(real(TR),[1,3,2])); % q x nm x m x s
end

% |==== Compose results Table in above format
for i = 1:nm
    rows = [];
    for k = 1:m
        for j = 1:nq
            rows = [rows, strcat("M",string(k),"_q",string(j))];
        end
    end
    
    Tables.(model_names(i)).TR2 = array2table(TR2(:,:,i),'VariableNames',cellstr(files),'RowNames',cellstr(rows));
end

%% |==== Trace R2 Only (aggregated and other versions)
%  -------------------
% Performance aggregated across quantile levels:
% Table 1):
%                            Trace R2
%          (50,50)  (50,100) ....        (200,200) 
%          ---------------------------------------
%    mod1 |
% M1 mod2 |
%    .... |
%    mod1 |
% M2 mod2 | 
%    ....
%
% Table 2):
%                            Trace R2
%                 q1             ...        qn    
%          (50,50) ... (200,200) ... (50,50) ... (200,200) 
%          -----------------------------------------------
%    mod1 |
% M1 mod2 |
%    .... |
%    mod1 |
% M2 mod2 | 
%    ....


model_select = ["CDG", "VBQFA"];
mod_id = [];
for i = 1:length(model_select)
   mod_id = [mod_id, find(model_names == model_select(i))]; 
end

TR2 = [];
TR3 = [];
for file = 1:length(files)
    load(strcat("MONTE_CARLO_",files(file),".mat"))
    
    % data for tables
    TR2 = cat(1,TR2,mean(real(TR(:,:,mod_id)),1));%cat(1,TR2,permute(mean(real(TR),1),[3,2,1]));
    TR3 = cat(1,TR3,reshape(permute(real(TR(:,:,mod_id)),[2,1,3]),1,m,nq,length(model_select))); % s x m x nq x nm 
    %TR3 = cat(1,TR3,reshape(permute(real(TR(:,:,mod_id)),[2,1,3]),1,[],length(mod_id)));
end
TR2 = reshape(permute(TR2,[3,2,1]),length(mod_id)*m,s,1);      %(m x nm) * s        
TR3 = reshape(permute(TR3,[4,2,1,3]),length(mod_id)*m,s*nq,1); %(m x nm) * (s x nq)

Tables.aggregated.TR_mean = TR2;
Tables.aggregated.TR      = TR3;

%% |==== CDG Style tables R2
%  -------------------------

% For each Model:
% Table 1) Indiv. R2 only: 
%                    q1           q2           q3 
%                 f1 f2 f3     f1 f2 f3     f1 f2 f3 
%             ---------------------------------------
%    (50,50)  |
% M1 (50,100) |
%    (50,200) |
%    (100,50) |
%    (100,100)|
%     ...
%


R2_ = []; % [s x (f*q) x m x nm]
TR_ = [];
m   = 8;
for file = 1:length(files)
    load(strcat("MONTE_CARLO_",files(file),".mat"))
    
    % compute mean Across MC samples 
    r2 = permute(squeeze(mean(R2,1)),[1,2,3,4]); % f, q, dgp, model
    r2 = reshape(r2,1,nq*r,m,nm);
    % data for tables
    R2_ = cat(1,R2_,r2);
    TR_ = cat(1,TR_,reshape(real(TR),1,nq,m+2,nm));
end

R2_ = reshape(permute(R2_,[1,3,2,4]),[],r*nq,nm); %(m*s) x (f*q) x nm
TR_ = reshape(permute(TR_,[1,3,2,4]),[],nq,nm);

% |==== Table 1):
%--------------------------------------------------------------------------

% create row and column names
rows = [];
for k = 1:m
    for j = 1:s
        rows = [rows, strcat("M",string(k),"_",files(j))];
    end
end

columns = [];
for k = 1:nq
    for j = 1:r
        columns = [columns, strcat("q",string(k),"_f",string(j))];
    end
end

% compose table
for i = 1:nm   
    Tables.(model_names(i)).R2_1 = array2table(R2_(:,:,i),'VariableNames',cellstr(columns),'RowNames',cellstr(rows));
end



%% |==== CDG Style tables MSD
%  -------------------------

% For each Model:
% Indiv. MSD only: 
%                    q1           q2           q3 
%                 f1 f2 f3     f1 f2 f3     f1 f2 f3 
%             ---------------------------------------
%    (50,50)  |
% M1 (50,100) |
%    (50,200) |
%    (100,50) |
%    (100,100)|
%     ...

MSD_ = []; % [s x (f*q) x m x nm]
for file = 1:length(files)
    load(strcat("MONTE_CARLO_",files(file),".mat"))
    
    % compute mean Across MC samples 
    msd = permute(squeeze(mean(real(MSD),1)),[1,2,3,4]); % f, q, dgp, model
    msd = reshape(msd,1,nq*r,m,nm);
    % data for tables
    MSD_ = cat(1,MSD_,msd);
end

MSD_ = reshape(permute(MSD_,[1,3,2,4]),[],r*nq,nm); %(m*s) x (f*q) x nm

% create row and column names
rows = [];
for k = 1:m
    for j = 1:s
        rows = [rows, strcat("M",string(k),"_",files(j))];
    end
end

columns = [];
for k = 1:nq
    for j = 1:r
        columns = [columns, strcat("q",string(k),"_f",string(j))];
    end
end

% compose table
for i = 1:nm   
    Tables.(model_names(i)).MSD = array2table(MSD_(:,:,i),'VariableNames',cellstr(columns),'RowNames',cellstr(rows));
end

save('MC_results_tables.mat','Tables')


%% Gen Histograms 

p = 1;
T = 10000;
probs = rand(T,1);

clear error
error(:,:,1)  = Normal(0,1^2,T,p);
error(:,:,2)  = trnd(3,T,p);                                       % CDD2021 DGP1
error(:,:,3)  = (abs(randn(T,1))*(randn(1,p)+1)).*trnd(3,T,p);     % CDD2021 DGP2
error(:,:,4)  = (probs<=1/5).*Normal(-22/25,1^2,T,p) + (probs<=2/5 & probs>1/5).*Normal(-49/125,(3/2)^2,T,p) + (probs>2/5).*Normal(49/250,(5/9)^2,T,p);
error(:,:,5)  = (probs>1/3).*Normal(0,1^2,T,p) + (probs<=1/3).*Normal(0,(1/10)^2,T,p);
error(:,:,6)  = (probs<=1/10).*Normal(0,1^2,T,p) + (probs>1/10).*Normal(0,(1/10)^2,T,p);
error(:,:,7)  = (probs<=1/2).*Normal(-1,(2/3)^2,T,p) + (probs>1/2).*Normal(1,(2/3)^2,T,p);
error(:,:,8)  = (probs<=1/2).*Normal(-3/2,(1/2)^2,T,p) + (probs>1/2).*Normal(3/2,(1/2)^2,T,p);
error(:,:,9)  = (probs>=3/4).*Normal(-43/100,1^2,T,p) + (probs<3/4).*Normal(107/100,(1/3)^2,T,p);
error(:,:,10) = (probs<=9/20).*Normal(-6/5,(3/5)^2,T,p) + (probs>9/20 & probs<=18/20).*Normal(6/5,(3/5)^2,T,p) + (probs>18/20).*Normal(0,(1/4)^2,T,p);

for i = 1:size(error,3)
    figure()
    histogram(error(:,:,i),100,'Normalization','pdf')
%     hold on
%       [f,xi] = ksdensity(error(:,:,i),'Function','pdf','Bandwidth', 0.1)
%       plot(xi,f,'LineWidth',2)
%     hold off
    %title(strcat('M',string(i)));
    orient(gcf,'landscape');
    print(gcf,[strcat('Figures\histogram_M', string(i))],'-depsc')
end

%% Other Chart ideas

Tables.aggregated.TR
type_ = ['o';'d';'*'];
xtick = ["(50,50)", "(50,100)", "(50,200)",...
        "(100,50)", "(100,100)", "(100,200)",...
        "(200,50)", "(200,100)", "(200,200)",...
        "(50,50)", "(50,100)", "(50,200)",...
        "(100,50)", "(100,100)", "(100,200)",...
        "(200,50)", "(200,100)", "(200,200)",...
        "(50,50)", "(50,100)", "(50,200)",...
        "(100,50)", "(100,100)", "(100,200)",...
        "(200,50)", "(200,100)", "(200,200)"];
% xtick = ["","(50,50)", "(50,100)", "(50,200)",...
%         "(100,50)", "(100,100)", "(100,200)",...
%         "(200,50)", "(200,100)",...
%         "(50,50)", "(50,100)", "(50,200)",...
%         "(100,50)", "(100,100)", "(100,200)",...
%         "(200,50)", "(200,100)",...
%         "(50,50)", "(50,100)", "(50,200)",...
%         "(100,50)", "(100,100)", "(100,200)",...
%         "(200,50)", "(200,100)"];

%1:3, 4:6

j = 1;
for i = 1:m
    dat = Tables.aggregated.TR(j:j+length(mod_id)-1,:);
    j = j+length(mod_id);
    
%     aux = repmat(NaN*dat',1,nq);
%     for q = 1:nq
%        aux(1+s*(q-1):s*q,(q-1)*length(mod_id)+1:length(mod_id)*q) = dat(:,1+s*(q-1):s*q)'; 
%     end
    
    figure();
    hold on
    for ii = 1:length(mod_id)
         plot(dat(ii,:)',type_(ii),'MarkerSize',8,'Linewidth',2)
    end

    xline(s+0.5,'Linewidth',2);
    xline(s*2+0.5,'Linewidth',2);
    set(gca,'XTickLabel',cellstr(xtick),'FontSize',8)
    set(gca,'XTick',[1:27])
    set(gca,'XLim',[0 28])
    xtickangle(45)
    annotation('textbox',[.24 .0 .1 .2],'String',"\tau_{0.25}",'EdgeColor','none','FontSize',12)
    annotation('textbox',[.51 .0 .1 .2],'String',"\tau_{0.50}",'EdgeColor','none','FontSize',12)
    annotation('textbox',[.755 .0 .1 .2],'String',"\tau_{0.75}",'EdgeColor','none','FontSize',12)
    pos = get(gca,'Position');
    set(gca,'Position',[pos(1), pos(2)+0.17, pos(3), 0.68])
    ylabel("R^2_{Trace}",'FontSize',10)    
    legend(["CDG", "VBQFA"],'Location','northoutside','Orientation','horizontal')
    hold off
    box on
    orient(gcf,'landscape');
    print(gcf,[strcat('Figures\dot_M', string(i))],'-depsc')
    
end

%% Misc
 
% TR_ = [];
% TR3 = [];
% for file = 1:length(files)
%     load(strcat("MONTE_CARLO_",files(file),".mat"))
%     TR_ = cat(1,TR_,reshape(real(TR),1,nm,nq,m));
%     % data for charts
%     TR3 = cat(4,TR3,permute(real(TR),[1,3,2])); % q x nm x m x s
% end
% 
% % build plot matrix
% select = [1,2,3,6];
% mat_ = [];
% for i = 1:size(TR3,3)
%     mat_ = [mat_; TR3(:,select,i); zeros(1,size(TR3(:,select,i),2))];
% end
% %mat_ = mat_./sum(mat_,2); 
% mat_ = mat_(1:end-1,:);
% b = barh(1:size(mat_,1),flipud(mat_),'stacked','Facecolor','flat');
% for k = 1:4
%          b(k).CData = k;
% end
% a = {'\tau = 0.9','M1 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M2 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M3 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M4 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M5 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M6 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M7 \tau = 0.5','\tau = 0.1',' ', ... 
%      '\tau = 0.9','M8 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M9 \tau = 0.5','\tau = 0.1',' ', ...
%      '\tau = 0.9','M10 \tau = 0.5','\tau = 0.1'}
% 
% %set(gca,'xticklabel',{'\tau = 0.1','\tau = 0.5','\tau = 0.9'})
% set(gca,'ytick',[1:size(mat_,1)])
% set(gca,'xtick',[0:4])
% set(gca,'yticklabel',a)'
% legend(model_names(select),'Location','southoutside','Orientation','horizontal')


% |==== Create
%Charts
% tiledlayout(1,s);
% %target DGP
% j = 1;
% l = 0;
% for i = 1:s%m*s
%     
% %     if l+1 >s
% %         l = 1;
% %         %j = j+1;
% %     else
% %         l = l+1;
% %     end
%     nexttile
%     
%     b = bar(TR3(:,:,j,i),'FaceColor','flat');
%     for k = 1:size(TR3(:,:,1,1),2)
%         b(k).CData = k;
%     end
%     set(gca,'xticklabel',{'\tau = 0.1','\tau = 0.5','\tau = 0.9'})
%     xlabel('Quantile Level')
%     ylabel('Trace R^2')
%     legend(model_names,'Location','southoutside','Orientation','horizontal')
% end