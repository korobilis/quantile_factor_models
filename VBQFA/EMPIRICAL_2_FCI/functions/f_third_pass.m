function [results] = f_third_pass(params, results)

   	if ~isfield(params, 'lag3p')
    	lag3p = 0;
   	else
    	lag3p = params.lag3p;
    end
    
    if ~isfield(params, 'trend3p') 
        trend = [ ];
        params.trend3p = 0;
    elseif isfield(params, 'trend3p') && params.trend3p == 0
        trend = [ ];
    elseif isfield(params, 'trend3p') && params.trend3p == 1
        if params.foretype == 2
            fprintf("VAR selected, no trend included!\n");
            trend = [ ];
        else
            trend = [1:params.T]';
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
    
    
    if params.FP_type == "QRFactors"
        params.k = size(results.F,1);
    end
    
    % univariate forecasts
    if params.foretype == 1 && (params.FP_type == 'VBQR' || params.FP_type == "VBQRTVP")    
        
        if max(size(params.quant))>1
            % initialize array to store forecasts
            forecasts = NaN(params.T-lag3p, size(params.Y,2), params.maxhorizon, max(size(params.quant)));
            betas     = NaN(params.k*(1+lag3p)+1+params.trend3p+arx3p+1, size(params.Y,2), params.maxhorizon, max(size(params.quant)));
        else
            forecasts = NaN(params.T-lag3p, size(params.Y,2), params.maxhorizon);
            betas     = NaN(params.k*(1+lag3p)+1+params.trend3p+arx3p+1, size(params.Y,2), params.maxhorizon);
        end
            
            for ii = 1:max(size(params.quant))
                FF = [ones(params.T,1), trend, lagmatrix(results.F(:,:,ii)',0:lag3p), lagmatrix(params.Y(:,arx3ptarget),0:arx3p)];          
                
                for i = 1:1:params.maxhorizon        
                    % adjust sample of outcome variables
                    Y = params.Y(i+1+lag3p:end,:);

                    % adjust sample of factors
                    F = FF(1+lag3p:end-i,:);

                    % compute forecast
                    beta = (F'*F)\(F'*Y);
                    betas(:,:,i,ii) = beta;
                    Yhat_aux = F*beta;

                    % store forecasts
                    forecasts(i+1:end,:,i,ii) = Yhat_aux;            
                end    
            end
            % compute forecast error
            ferr = params.Y(1+lag3p:end,:)-forecasts;

            % store results
            results.forecasts = forecasts;
            results.ferr      = ferr;
            results.betas     = betas;
            
        elseif params.foretype == 1 && (params.FP_type ~= 'VBQR' || params.FP_type ~= "VBQRTVP")     
            
            % initialize array to store forecasts
            forecasts = NaN(params.T-lag3p, size(params.Y,2), params.maxhorizon);
            betas     = NaN(params.k*(1+lag3p)+1+params.trend3p+arx3p+1, size(params.Y,2), params.maxhorizon); %NaN(params.k+params.addconstant, size(params.Y,2), params.maxhorizon);

            % create RHS variables and add constant
            %if params.addconstant == 1
                FF = [ones(params.T,1), trend, lagmatrix(results.F',0:lag3p), lagmatrix(params.Y(:,arx3ptarget),0:arx3p)];          
            %else
                %FF = results.F';
            %end

            for i = 1:1:params.maxhorizon        
                % adjust sample of outcome variables
                Y = params.Y(i+1+lag3p:end,:);

                % adjust sample of factors
                F = FF(1+lag3p:end-i,:);

                % compute forecast
                beta = (F'*F)\(F'*Y);
                betas(:,:,i) = beta;
                Yhat_aux = F*beta;

                % store forecasts
                forecasts(i+1:end,:,i) = Yhat_aux;            
            end

            % compute forecast error
            ferr = params.Y(1+lag3p:end,:)-forecasts;

            % store results
            results.forecasts = forecasts;
            results.ferr      = ferr;
            results.betas     = betas;
    
    elseif params.foretype == 2 && (params.FP_type == 'VBQR' || params.FP_type == "VBQRTVP")    
       
        forecasts = NaN(params.T-lag3p, size(params.Y,2), params.maxhorizon);
        betas     = NaN(size(results.F,3)*(1+lag3p)+1+params.trend3p+arx3p+1, size(params.Y,2), params.maxhorizon);
        
        ftemp = (permute(results.F,[2,1,3]));
        ftemp = ftemp(:,:);
       
        FF = [ones(params.T,1), trend, lagmatrix(ftemp(:,:),0:lag3p), lagmatrix(params.Y(:,arx3ptarget),0:arx3p)];          

        for i = 1:1:params.maxhorizon        
            % adjust sample of outcome variables
            Y = params.Y(i+1+lag3p:end,:);

            % adjust sample of factors
            F = FF(1+lag3p:end-i,:);

            % compute forecast
            beta = (F'*F)\(F'*Y);
            betas(:,:,i) = beta;
            Yhat_aux = F*beta;

            % store forecasts
            forecasts(i+1:end,:,i) = Yhat_aux;            
        end    
        
        % compute forecast error
        ferr = params.Y(1+lag3p:end,:)-forecasts;

        % store results
        results.forecasts = forecasts;
        results.ferr      = ferr;
        results.betas     = betas;
            
     elseif params.foretype == 3 && (params.FP_type == 'VBQR' || params.FP_type == "VBQRTVP")    
        if max(size(params.quant))>1
            % initialize array to store forecasts
            forecasts = NaN(params.T-lag3p, size(params.Y,2), params.maxhorizon, max(size(params.quant)));
            betas     = NaN(size(results.F,3)*(1+lag3p)+1+params.trend3p+arx3p+1, size(params.Y,2), params.maxhorizon, max(size(params.quant)));
        else
            forecasts = NaN(params.T-lag3p, size(params.Y,2), params.maxhorizon);
            betas     = NaN(size(results.F,3)*(1+lag3p)+1+params.trend3p+arx3p+1, size(params.Y,2), params.maxhorizon);
        end
        
        ftemp = (permute(results.F,[2,1,3]));
        ftemp = ftemp(:,:);
       
        FF = [ones(params.T,1), trend, lagmatrix(ftemp(:,:),0:lag3p), lagmatrix(params.Y(:,arx3ptarget),0:arx3p)];
         
        for i = 1:1:params.maxhorizon        
            % adjust sample of outcome variables
            Y = params.Y(i+1+lag3p:end,:);

            % adjust sample of factors
            F = FF(1+lag3p:end-i,:);

            % compute forecast
            [beta,~,~] = VBTVPQR(Y,F,200,params.quant,0);
            %[beta,~,~,~,~,~,~] = VBQR(Y,F,100,params.quant);
            
            %
            Yhat_aux = F*squeeze(beta(end,:,:));
            
            betas(:,:,i,:) = squeeze(beta(end,:,:)); 
            
            % store forecasts
            forecasts(i+1:end,:,i,:) = Yhat_aux;            
         end    

        % compute forecast error
        ferr = params.Y(1+lag3p:end,:)-forecasts;

        % store results
        results.forecasts = forecasts;
        results.ferr      = ferr;
        results.betas     = betas;
        
%     elseif params.foretype == 2   % BVAR forecasts
%         
%         % initialize object storing the forecasts
%         forecasts = zeros(size(params.Y,2), params.maxhorizon);
%         
%         % run VAR and compute forecasts
%         FF = lagmatrix(results.F',0:lag3p);
%         
%         [Y_pred] = BVAR_CCM([params.Y(1+lag3p:end,:), FF(1+lag3p:end,:)],params.VARlags,params.maxhorizon);
%         
%         for i = 1:size(params.Y,2)
%             
%             % store forecasts for variables of interest
%             forecasts(i,:) = Y_pred(i,:);
%         end
%         
%         % store results
%         results.forecasts = forecasts;
    end
    
end

