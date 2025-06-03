function [results] = f_text_3PRF(params)
    
    % if verbose option not set, set verbose to false
    if ~isfield(params, 'verbose')
        params.verbose = 0;
    end

    if params.forecasts   == 0
        
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
        
        % initialize output array
        results_.forecasts = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1);
        results_.ref = NaN(params.maxhorizon, size(params.Y,2), max(size(params.dates))-id_+1);
        
        % initialize outputs for loadings and factors
        results_.F   = NaN(params.k, params.T, max(size(params.dates))-id_+1);
        
        % initialize output for loadings
        %if params.FP_type == "OLS"
            %results_.phi_hat = NaN(params.k+params.addconstant, params.p, max(size(params.dates))-id_);
        %elseif params.FP_type == "Forgetting Factor" || params.FP_type == "TVP Lasso" || params.FP_type == "Variational Bayes"
            %results_.phi_hat = NaN(params.T, params.k+params.addconstant, params.p, max(size(params.dates))-id_);
        %end
        
        for i = id_:max(size(params.dates))
            
            fprintf('Completed %s percent...\n',string((i-id_)/(max(size(params.dates))-id_)*100))
            
            %adjust input arguments
            iterator.T = i-1;
            iterator.Y = params.Y(1:i-1,:);
            iterator.Z = params.Z(1:i-1,:);
            iterator.dtm = params.dtm(1:i-1,:);
            
            % compute the first pass
            results = f_first_pass(iterator);

            % compute the second pass
            results = f_second_pass(iterator, results);

            % compute the third pass
            results = f_third_pass(iterator, results);            
            
            if params.foretype == 1  
                % compose RHS variables and add constant
                FF = [ones(1,1); full(results.F(:,end))];         
                
                for h = 1:params.maxhorizon
                    results_.forecasts(h,:,i-id_+1) = FF'*results.betas(:,:,h);
                end
            elseif params.foretype == 2 
                
                % store VAR forecasts
                results_.forecasts(:,:,i-id_+1) = results.forecasts';
            end
            
            % store factor loadings
            %if params.FP_type == "OLS"
            %    results_.phi_hat(:, :, i-id_+1) = results.phi_hat;
            %elseif params.FP_type == "Forgetting Factor" || params.FP_type == "TVP Lasso" || params.FP_type == "Variational Bayes"
            %    results_.phi_hat(1:size(results.phi_hat,1), :, :, i-id_+1) = results.phi_hat;
            %end
            
            % storing factor estimates for each vintage
            results_.F(:, 1:size(results.F,2), i-id_+1) = results.F;     
            
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
        %results.ferr_min = min(results.ferr,[],3);
        %results.ferr_max = max(results.ferr,[],3);
        %results.ferr_prc = prctile(results.ferr,[0.25, 0.75],3);
    end
end

