function [results] = f_first_pass(params)

    % adjust series for missing values
    params.Z = params.Z(~isnan(params.Z),:);
    params.T = size(params.Z,1);
    
    if params.addconstant == 1 
        % add a constant to the matrix of factor proxies
        Z_sp = sparse([ones(params.T,1), params.Z]);
    else 
        Z_sp = sparse(params.Z);
    end
    
    if params.FP_type == "VBQR"
        
        if isfield(params, 'leads')
            %fprintf("Detected field 'leads': Will account for leads in first pass!\n");
            
            % adjust input data
            Y    = params.dtm(1:params.T-params.leads,:);
            Z_sp = Z_sp(1+params.leads:end,:); 
        else
            Y = params.dtm;
        end
        
        if max(size(params.quant))>1
            phi_hat = zeros(size(Z_sp,2),size(Y,2),max(size(params.quant)));
        else
            phi_hat = zeros(size(Z_sp,2),size(Y,2));
        end
        
        for i = 1:size(Y,2)
           ids = ~[isnan(Y(:,i))+sum(isnan(Z_sp),2)];
           if params.verbose == 0
                if mod(i,10) == 0
                    fprintf('Iteration %d...\n',i);
                end
           end
           %if i == 1
           %    fprintf(' ')
           %end
           [beta,~,~] = VBTVPQR(Y(ids,i),Z_sp(ids,:),100,params.quant,0);
           %[beta,~,~,~,~,~,~] = VBQR(Y(ids,i),Z_sp(ids,:),100,params.quant);
           % precompute regression matrices
           %XpXi = Z_sp(ids,:)'*Z_sp(ids,:);
           %XpY  = Z_sp(ids,:)'*Y(ids,i);
           %phi_hat(:,i) = XpXi\XpY;
           phi_hat(:,i,:) = reshape(beta(end,:,:),size(beta(end,:,:),2),1,max(size(params.quant))); 
        end
    
    elseif params.FP_type == "VBQRTVP"
        
        if isfield(params, 'leads')
            %fprintf("Detected field 'leads': Will account for leads in first pass!\n");
            
            % adjust input data
            Y    = params.dtm(1:params.T-params.leads,:);
            Z_sp = Z_sp(1+params.leads:end,:); 
        else
            Y = params.dtm;
        end
        
        % initialize containers 
        if max(size(params.quant))>1
            phi_hat = zeros(params.T,size(Z_sp,2),size(Y,2),max(size(params.quant)));
        else
            phi_hat = zeros(params.T,size(Z_sp,2),size(Y,2));
        end
        
        for i = 1:size(Y,2)
           ids = ~[isnan(Y(:,i))+sum(isnan(Z_sp),2)];
           if params.verbose == 0
                if mod(i,10) == 0
                    fprintf('Iteration %d...\n',i);
                end
           end
           %if i == 1
           %    fprintf(' ')
           %end
           [beta,~,~] = VBTVPQR(Y(ids,i),Z_sp(ids,:),100,params.quant,3);
           %[beta,~,~,~,~,~,~] = MC_VBTVPQR(Y(ids,i),Z_sp(ids,:),100,params.quant,params);

           phi_hat(:,:,i,:) = reshape(beta,size(beta,1),size(beta,2),max(size(params.quant))); 
        end
    
    elseif params.FP_type == "QRFactors"
        
        if params.addconstant == 0
            params.addconstant = 1;
        end
        
        %x = [ones(params.T-1,1), params.Y(1:end-1)];
        x = [ones(params.T,1)];
        
        %[beta,~,~] = VBTVPQR(params.Y(2:end),x,300,params.quant,1);
        [beta,~,~] = VBTVPQR(params.Y,x,300,params.quant,1);

        %[beta,~,~,~,~,~,~] = MC_VBTVPQR(params.Y,x,200,params.quant,params);
        
        params.Z = beta(:,:);
        params.k = size(params.Z,2);
        Z_sp = sparse(params.Z);
        
        if isfield(params, 'leads')
            %fprintf("Detected field 'leads': Will account for leads in first pass!\n");
            
            % adjust input data
            Y    = params.dtm(1:params.T-params.leads,:);
            Z_sp = Z_sp(1+params.leads:end,:); 
        else
            Y = params.dtm;
        end
        
        phi_hat = zeros(size(Z_sp,2),size(Y,2));
        
        for i = 1:size(Y,2)
           ids = ~[isnan(Y(:,i))+sum(isnan(Z_sp),2)];
           
           %fprintf('%i',i)
           
           % precompute regression matrices
           XpXi = Z_sp(ids,:)'*Z_sp(ids,:);
           XpY  = Z_sp(ids,:)'*Y(ids,i);
           phi_hat(:,i) = XpXi\XpY;
        end
        
    elseif params.FP_type == "OLS"
    
        if isfield(params, 'leads')
            %fprintf("Detected field 'leads': Will account for leads in first pass!\n");
            
            % adjust input data
            Y    = params.dtm(1:params.T-params.leads,:);
            Z_sp = Z_sp(1+params.leads:end,:); 
        else
            Y = params.dtm;
        end
        
        phi_hat = zeros(size(Z_sp,2),size(Y,2));
        
        for i = 1:size(Y,2)
           ids = ~[isnan(Y(:,i))+sum(isnan(Z_sp),2)];
           
           % precompute regression matrices
           XpXi = Z_sp(ids,:)'*Z_sp(ids,:);
           XpY  = Z_sp(ids,:)'*Y(ids,i);
           phi_hat(:,i) = XpXi\XpY;
        end
        % precompute regression matrices
        %XpXi = Z_sp'*Z_sp;
        %XpY  = Z_sp'*Y;

        % compute OLS coefficients
        %phi_hat = XpXi\XpY;
    
    elseif params.FP_type == "OLS restricted"
        
        if isfield(params, 'leads')
            %fprintf("Detected field 'leads': Will account for leads in first pass!\n");
            
            % adjust input data
            Y    = params.dtm(1:params.T-params.leads,:);
            Z_sp = Z_sp(1+params.leads:end,:); 
        else
            Y = params.dtm;
        end
        
        if params.addconstant == 1
           
           % intialize the output array for the coefficients 
           phi_hat = zeros(size(Z_sp,2),size(params.dtm,2)); 
           
           % create and auxiliary matrix to store the relevant coefficients
           phi_    = zeros(size(Z_sp,2)-1,2,size(params.dtm,2));
           for i = 1+params.addconstant:size(Z_sp,2) 
               % precompute regression matrices
               ZZ = Z_sp(:,[1, i]);
               XpXi = ZZ'*ZZ;
               XpY  = ZZ'*Y;
               
               % compute OLS coefficients
               phi_(i-1,:,:) = XpXi\XpY;

           end
           
           for i = 1:size(params.dtm,2)
               
               [~, id] = max(abs(phi_(:,2,i)));
               phi_hat([1,id+1],i) = phi_(id,:,i);
                 
           end    
       
        else
            
           % intialize the output array for the coefficients 
           phi_hat = zeros(size(Z_sp,2),size(params.dtm,2)); 
            
           % create and auxiliary matrix to store the relevant coefficients
           phi_    = zeros(size(Z_sp,2),size(params.dtm,2));
           for i = 1:size(Z_sp,2) 
               % precompute regression matrices
               ZZ = Z_sp(:, i);
               XpXi = ZZ'*ZZ;
               XpY  = ZZ'*Y;
               
               % compute OLS coefficients
               phi_(i,:) = XpXi\XpY;

           end
           
           for i = 1:size(params.dtm,2)
               
               [~, id] = max(abs(phi_(:,i)));
               phi_hat(id,i) = phi_(id,i);
                 
           end    
            
        end
        
        % convert phi_hat to sparse
        phi_hat = sparse(phi_hat);

    elseif params.FP_type == "Hard Thresholding"
       
        if isfield(params, 'leads')
            %fprintf("Detected field 'leads': Will account for leads in first pass!\n");
            
            % adjust input data
            Y    = params.dtm(1:params.T-params.leads,:);
            Z_sp = Z_sp(1+params.leads:end,:); 
        else
            Y = params.dtm;
        end
        
        if params.addconstant == 1
            
           % intialize the output array for the coefficients 
           phi_hat = zeros(size(Z_sp,2),size(params.dtm,2));  
           
           % create and auxiliary matrix to store the relevant coefficients
           phi_    = zeros(size(Z_sp,2)-1,2,size(params.dtm,2)); 
           p_vals  = zeros(size(Z_sp,2)-1,size(params.dtm,2));
           
           %
           
           for i = 1+params.addconstant:size(Z_sp,2) 
               % precompute regression matrices
               ZZ = Z_sp(:,[1, i]);
               XpXi = ZZ'*ZZ;
               XpY  = ZZ'*Y;
               
               % compute OLS coefficients
               phi_(i-1,:,:) = XpXi\XpY;
               
               % back out the residuals
               res = Y-ZZ*squeeze(phi_(i-1,:,:));
               
               % compute residual variance;
               s_hat = 1/(size(params.dtm,1)-1)*diag(res'*res);
               
               % compute the standard errors
               invXpXi = inv(XpXi);
               se = sqrt(invXpXi(2,2).*s_hat);
               
               % compute t-stat
               tstat = abs(squeeze(phi_(i-1,2,:))./se);
               
               % compute the two sided p-values
               p_vals(i-1,:) = 2*(1-normcdf(tstat));
           end
           
           % extract coeff for given significance level
           alpha = 0.05;
           id = p_vals<=alpha;
           
           % run regression with constant
           for i = 1:size(params.dtm,2)
               if sum(id(:,i)~=0)>0
                   ZZ = Z_sp(:,boolean([1, id(:,i)']));
                   
                   XpXi = ZZ'*ZZ;
                   XpY  = ZZ'*Y(:,i);
               
                   % compute OLS coefficients
                   XpXi\XpY;
                   
                   phi_hat(boolean([1, id(:,i)']),i) = XpXi\XpY;
               end
           end

        else
           % intialize the output array for the coefficients 
           phi_hat = zeros(size(Z_sp,2),size(params.dtm,2)); 
            
           % create and auxiliary matrix to store the relevant coefficients
           phi_    = zeros(size(Z_sp,2),size(params.dtm,2));
           p_vals  = zeros(size(Z_sp,2),size(params.dtm,2));
           
           for i = 1:size(Z_sp,2) 
               % precompute regression matrices
               ZZ = Z_sp(:, i);
               XpXi = ZZ'*ZZ;
               XpY  = ZZ'*Y;
               
               % compute OLS coefficients
               phi_(i,:) = XpXi\XpY;
               
               % back out the residuals
               res = Y-ZZ*phi_(i,:);
               
               % compute residual variance;
               s_hat = 1/(size(params.dtm,1)-1)*diag(res'*res);
               
               % compute the standard errors
               se = sqrt(XpXi\s_hat);
               
               % compute t-stat
               tstat = abs(phi_(i,:)./se');
               
               % compute the two sided p-values
               p_vals(i,:) = 2*(1-normcdf(tstat));
           end
           
           % extract coeff for given significance level
           alpha = 0.05;
           id = p_vals<=alpha;
           
           %phi_hat = phi_.*id;
           for i = 1:size(params.dtm,2)
               if sum(id(:,i)~=0)>0
                   ZZ = Z_sp(:,boolean(id(:,i)'));
                   
                   XpXi = ZZ'*ZZ;
                   XpY  = ZZ'*Y(:,i);
               
                   % compute OLS coefficients
                   XpXi\XpY;
                   
                   phi_hat(boolean(id(:,i)'),i) = XpXi\XpY;
               end
           end
                      
        end
        
        % convert phi_hat to sparse
        phi_hat = sparse(phi_hat);
        
               
    elseif params.FP_type == "Forgetting Factor"
        
        % construct the prior given the input arguments
        prior = f_construct_prior_template(params, params.k+params.addconstant);
        
        % check whether values for prior set
        if isfield(params, 'priors')
            
            % overwrite default for lambda, if value set
            if isfield(params.priors, 'lambda')
                prior.lambda = params.priors.lambda;
            end
            
            % overwrite default for kappa, if value set
            if isfield(params.priors, 'kappa')
                prior.kappa = params.priors.kappa;
            end
        end
        
        % initialize structure to store the coefficients
        phi_hat = zeros(params.T, params.k+params.addconstant, params.p);
        
        % initialize fields if dynamic coefficients activated
        if params.priors.forgetting == 2
            lambda_t = zeros(params.T, params.p);
            kappa_t = zeros(params.T, params.p);
        end
        

        if params.verbose == 0
            disp('Estimate TVP regression using Kalman filter')
        end
        
        % iterate over terms in the document-term matrix 
        for i = 1:1:params.p
            
            if params.verbose == 0
                if mod(i,100) == 0
                    fprintf('At iteration %d...\n',i);
                end
            end
            
            output = f_TVP_forgetting_factors(params, params.dtm(:,i), Z_sp, params.k+params.addconstant, prior);
            phi_hat(:,:,i) = output.beta;
            
            % if dynamic factors activated, save results
            if params.priors.forgetting == 2
                lambda_t(:,i) = output.lambda_t;
                kappa_t(:,i) = output.kappa_t;
            end
        end
        
        % store forgetting factors, if dynamic option activated
        if params.priors.forgetting == 2
            results.lambda_t = lambda_t;
            results.kappa_t = kappa_t;
        end
        
    elseif params.FP_type == "TVP Lasso"    
        
        % construct the prior given the input arguments
        prior = f_construct_prior_template(params, params.k+params.addconstant);
        
        % check whether values for prior set
        if isfield(params, 'priors')
            
            % overwrite default for number of CV samples, if value set
            if isfield(params.priors, 'CV')
                prior.CV = params.priors.CV;
            end
           
        end
        
        % initialize structure to store the coefficients
        phi_hat = zeros(params.T, params.k+params.addconstant, params.p);
        
        if params.verbose == 0
            disp('Estimate TVP regression using lasso')
        end
        
        % iterate over terms in the document-term matrix 
        parfor i = 1:params.p % parfor
                        
            fprintf('At iteration %d...\n',i);
              
            %beta_lasso = f_TVP_lasso(params, full(params.dtm(:,i)), Z_sp, params.k+params.addconstant, prior);
            beta_lasso = f_TVP_lasso_new(full(params.dtm(:,i)), Z_sp, prior);
            
            
            phi_hat(:,:,i) = beta_lasso;
            
        end
        
    elseif params.FP_type == "Variational Bayes"
        % construct the prior given the input arguments
        prior = f_construct_prior_template(params, params.k+params.addconstant);
        
        % initialize structure to store the coefficients
        phi_hat = zeros(params.T, params.k+params.addconstant, params.p);
        
        if params.verbose == 0
            disp('Estimation using Variational Bayes')
        end
        
        % iterate over terms in the document-term matrix 
        parfor i = 1:params.p % parfor
                        
            fprintf('At iteration %d...\n',i);
              
            %beta_lasso = f_TVP_lasso(params, full(params.dtm(:,i)), Z_sp, params.k+params.addconstant, prior);
            beta_VB = f_VBDVS(full(params.dtm(:,i)), Z_sp, prior);
            
            
            phi_hat(:,:,i) = beta_VB;
            
        end
        
        
    end
    
    % store results
    results.phi_hat = phi_hat;
    

end

