function [params] = f_transform(params, transformation, scaling)

    % data transformations 
    if strcmp(transformation,'FD')
        % compute first differences
        params.dtm   = params.dtm(2:end,:)-params.dtm(1:end-1,:);
        
        % adjust length of dates vector
        params.dates = params.dates(2:end,:);
        
        % adjust Y and proxies
        params.Z = table2array(params.Z(params.dates,:));           
        params.Y = table2array(params.Y(params.dates,:)); 
        params.T = size(params.dates,1);
        
        % standardize the factor proxies
        params.Z = (params.Z-mean(params.Z))./std(params.Z);
        
    elseif strcmp(transformation,'AD')
        % compute annual differences
        params.dtm = params.dtm(13:end,:)-params.dtm(1:end-12,:);
        
        % adjust length of dates vector
        params.dates = params.dates(13:end);
        
        % adjust Y and proxies
        params.Z = table2array(params.Z(params.dates,:));           
        params.Y = table2array(params.Y(params.dates,:)); 
        params.T = size(params.dates,1);
        
        % standardize the factor proxies
        params.Z = (params.Z-mean(params.Z))./std(params.Z);
        
    elseif strcmp(transformation,'DS')
        % create seasonal dummies
        X = month(params.dates) == 1:12;
        
        % se-seasonalize dtm matrix
        params.dtm = sparse(params.dtm-X*((X'*X)\X'*params.dtm));
        
        % adjust Y and proxies
        params.Z = table2array(params.Z(params.dates,:));           
        params.Y = table2array(params.Y(params.dates,:)); 
        params.T = size(params.dates,1);
        
        % standardize the factor proxies
        params.Z = (params.Z-mean(params.Z))./std(params.Z);
    else
        % adjust Y and proxies
        params.Z = table2array(params.Z(params.dates,:));           
        params.Y = table2array(params.Y(params.dates,:)); 
        params.T = size(params.dates,1);
        
    end
    
        % standardize the factor proxies
        params.Z = (params.Z-mean(params.Z))./std(params.Z);
    
    % data scaling
    if strcmp(scaling,'Zscore')
        params.dtm = (params.dtm-mean(params.dtm))./std(params.dtm);
    end

end


