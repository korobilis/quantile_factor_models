function [results] = f_text_3PRF_leads(params)
    
    % check datatype of input sequences
    if istimetable(params.Z)
        params.Z = table2array(params.Z);
        if isfield(params,'standardize') && params.standardize == 1
            params.Z = normalize(params.Z,'zscore');
        end
    end
    if istimetable(params.dtm)
        params.dtm = table2array(params.dtm);
        if isfield(params,'standardize') && params.standardize == 1
            params.dtm = normalize(params.dtm,'zscore');
        end
    end
    if istimetable(params.Y)
        params.Y = table2array(params.Y);
    end

    % if verbose option not set, set verbose to false
    if ~isfield(params, 'verbose')
        params.verbose = 0;
    end

    if ~isfield(params, 'lag3p')
    	lag3p = 0;
   	else
    	lag3p = params.lag3p;
    end
    
    if isfield(params, 'rolling')
        if strcmp(params.FP_type,'OLS')
            if ~isfield(params,'window') && params.rolling == 1
                error('No window size set!')
            end
        else
            warning("rolling only allowed for 'OLS'. Turned off rolling window.")
            params.rolling = 0;
        end
    end
    
    if ~isfield(params, 'arx3p') 
        arx3p = -1;
        arx3ptarget = [];
    elseif isfield(params, 'arx3p') && params.foretype == 2
        fprintf("VAR selected, no additional lags of dependent variable included!\n");
        arx3p = -1;
    else
        arx3p = 0;
        arx3ptarget = params.arx3ptarget;
    end
    
    if params.forecasts   == 0
        
        if isfield(params, 'leads')
            fprintf("Detected field 'leads': Will account for leads in first pass!\n");
        end
        
        % compute the first pass
        results = f_first_pass(params);

        % compute the second pass
        results = f_second_pass(params, results);

        % compute the third pass
        results = f_third_pass(params, results);
    
    elseif params.forecasts == 1
        
        if ~isfield(params, 'forestart')
            error('No start date for out-of-sample forecasts set. Set datetime in params.forestart!')
        end
        
        % identify index of forecast start date
        id_ = find(params.forestart == params.dates);
        
        % intialize input arguments 
        iterator = params;
        
        if params.FP_type == "QRFactors"
            qq = 1;
            pd = max(size(params.quant));
        else
        	qq = max(size(params.quant));
            pd = params.k;
        end
        
        % initialize output array
        if params.foretype == 1 && qq>1
            results_.forecasts = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1,qq);
            results_.ref = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1);
            
            % initialize outputs for loadings and factors
            results_.F   = NaN(pd, params.T, max(size(params.dates))-id_+1,qq);
        
        elseif params.foretype == 3 && qq>1
            
            results_.forecasts = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1,qq);
            results_.ref = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1);
            
            % initialize outputs for loadings and factors
            results_.F   = NaN(pd, params.T, max(size(params.dates))-id_+1,qq);
        else
            results_.forecasts = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1);
            results_.ref = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1);
            
            % initialize outputs for loadings and factors
            results_.F   = NaN(pd, params.T, max(size(params.dates))-id_+1);
        end

        
        % initialize output for loadings
        %if params.FP_type == "OLS"
            %results_.phi_hat = NaN(params.k+params.addconstant, params.p, max(size(params.dates))-id_);
        %elseif params.FP_type == "Forgetting Factor" || params.FP_type == "TVP Lasso" || params.FP_type == "Variational Bayes"
            %results_.phi_hat = NaN(params.T, params.k+params.addconstant, params.p, max(size(params.dates))-id_);
        %end
        
        if isfield(params, 'leads')
            fprintf("Detected field 'leads': Will account for leads in first pass!\n");
        end
        
        for i = id_:max(size(params.dates))
            
            fprintf('Completed %s percent...\n',string((i-id_)/(max(size(params.dates))-id_)*100))
            
            if ~isfield(params, 'rolling') || params.rolling == 0
                %adjust input arguments
                iterator.T = i;
                iterator.Y = params.Y(1:i,:);
                iterator.Z = params.Z(1:i,:);
                iterator.dtm = params.dtm(1:i,:);
            elseif isfield(params, 'rolling') && params.rolling == 1
                %adjust input arguments
                iterator.T = params.window;
                iterator.Y = params.Y(i-params.window+1:i,:);
                iterator.Z = params.Z(i-params.window+1:i,:);
                iterator.dtm = params.dtm(i-params.window+1:i,:);
            end
            
            if ~isfield(params, 'trend3p') 
                trend = [ ];
            elseif isfield(params, 'trend3p') && params.trend3p == 0
                trend = [ ];
            elseif isfield(params, 'trend3p') && params.trend3p == 1
                if params.foretype == 2
                    fprintf("VAR selected, no trend included!\n");
                    trend = [ ];
                else
                    trend = [1:iterator.T]';
                end
            end
            
%             iterator.T = i;
%             iterator.Y = params.Y(1:i,:);
%             iterator.Z = params.Z(1:i,:);
%             iterator.dtm = params.dtm(1:i,:);
            
            % compute the first pass
            results = f_first_pass(iterator);

            % compute the second pass
            results = f_second_pass(iterator, results);

            % compute the third pass
            results = f_third_pass(iterator, results);            
            
            if params.foretype == 1  
                
                % compose RHS variables and add constant
                %FF = [ones(1,1); full(results.F(:,end))];
                for ii = qq
                    FF = [ones(iterator.T,1), trend, lagmatrix(results.F(:,:,ii)',0:lag3p), lagmatrix(iterator.Y(:,arx3ptarget),0:arx3p)]; 
                    for h = 1:params.maxhorizon
                        results_.forecasts(h,:,i-id_+1,ii) = full(FF(end,:))*results.betas(:,:,h);
                    end
                end
            elseif params.foretype == 2
                 ftemp = (permute(results.F,[2,1,3]));
                 ftemp = ftemp(:,:);
                 FF = [ones(iterator.T,1), trend, lagmatrix(ftemp,0:lag3p), lagmatrix(iterator.Y(:,arx3ptarget),0:arx3p)]; 
                 for h = 1:params.maxhorizon
                     results_.forecasts(h,:,i-id_+1) = full(FF(end,:))*results.betas(:,:,h);
                 end
            elseif params.foretype == 3
                 ftemp = (permute(results.F,[2,1,3]));
                 ftemp = ftemp(:,:);
                 FF = [ones(iterator.T,1), trend, lagmatrix(ftemp,0:lag3p), lagmatrix(iterator.Y(:,arx3ptarget),0:arx3p)]; 
                 for ii = 1:max(size(params.quant))
                    for h = 1:params.maxhorizon
                         results_.forecasts(h,:,i-id_+1,ii) = full(FF(end,:))*results.betas(:,:,h,ii);
                    end
                 end
            %elseif params.foretype == 2 
                
                % store VAR forecasts
                %results_.forecasts(:,:,i-id_+1) = results.forecasts';
            end
            
           
            % store factor loadings
            %if params.FP_type == "OLS"
            %    results_.phi_hat(:, :, i-id_+1) = results.phi_hat;
            %elseif params.FP_type == "Forgetting Factor" || params.FP_type == "TVP Lasso" || params.FP_type == "Variational Bayes"
            %    results_.phi_hat(1:size(results.phi_hat,1), :, :, i-id_+1) = results.phi_hat;
            %end
            
            % storing factor estimates for each vintage
            results_.F(:, 1:size(results.F,2), i-id_+1,:) = results.F;     
            
            % collect reference time series
            results_.ref(1:params.maxhorizon+min(size(params.Y,1)-(i+params.maxhorizon),0),:,i-id_+1) = params.Y(i+1:i+params.maxhorizon+min(size(params.Y,1)-(i+params.maxhorizon),0),:);
            
        end
        
        % store results
        results.forecasts = results_.forecasts;
        results.F         = results_.F;
        %results.phi_hat   = results_.phi_hat;
        
        % overwrite zeros in last sheet and save reference values
        results_.ref(results_.ref==0) = NaN;
        results.ref = results_.ref;
        
        % compute forecast errors
        results.ferr     = results.ref-results.forecasts;
        results.RMSE     = sqrt(nanmean(results.ferr.^2,3));
        results.MAD      = nanmean(abs(results.ferr),3);
        
        if strcmp(params.FP_type,'VBQR') || strcmp(params.FP_type,'VBQRTVP') 
            results.Qscores = NaN(params.maxhorizon, size(params.Y,2), max(size(params.quant)));
            for jj = 1:max(size(params.quant))
                results.Qscores(:,:,jj)  = mean((results.ref - results.forecasts(:,:,:,jj)).*(params.quant(jj)-double(results.ref<=results.forecasts(:,:,:,jj))),3);     
            end
        end
            %results.ferr_min = min(results.ferr,[],3);
        %results.ferr_max = max(results.ferr,[],3);
        %results.ferr_prc = prctile(results.ferr,[0.25, 0.75],3);
    end
end

