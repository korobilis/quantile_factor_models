function [Xraw,dates,names] = load_data(startdate,enddate, file)
    
    %%% ====| Extract Raw Data
    data  = readtable(file);         % read in data
    data  = table2timetable(data);   % transform to timetable
    incl  = table2array(data(1,:));  % extract tcodes
    tcode = table2array(data(2,:));  % extract tcodes
    data(1:2,:)   = [];                % drop index columns
    data(:,~incl) = [];
    tcode(:,~incl) = [];
    data  = data(~isnat(data.Properties.RowTimes),:); % drop NaT rows
    
    %%% ====| Extract Metadata
    dates = data.Properties.RowTimes;
    names = data.Properties.VariableNames;
    
    %%% ====| Apply Transformations
    yt = prepare_missing(table2array(data),tcode);
    
    %%% ====| Restrict Sample
    start_id = find(dates==datetime(startdate));
    end_id   = find(dates==datetime(enddate));
    yt    = yt(start_id:end_id,:);
    dates = dates(start_id:end_id);
    
    %%% ====| Drop Series with Missing Values
    names = names(~any(isnan(yt),1));
    Xraw  = yt(:,~any(isnan(yt),1));
    [T,N] = size(Xraw);
    
    %%% ====| Standardize Predictors
%     % standardize the data
%     m = mean(Xraw);
%     v = var(Xraw);
% 	X = (Xraw-ones(T,1)*m)./(ones(T,1)*sqrt(v));
    
    
end

