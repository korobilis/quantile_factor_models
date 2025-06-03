function [results] = f_second_pass(params, results)
    
     if params.FP_type == "VBQR"
        
        % extract dtm
        X = params.dtm';
         
        if max(size(params.quant))>1
            F = zeros(size(results.phi_hat,1),size(X,2),max(size(params.quant))); 
        else
            F = zeros(size(results.phi_hat,1),size(X,2)); 
        end
        
        for ii = 1:max(size(params.quant))
            % extract the coefficients
            phi = results.phi_hat(:,:,ii)';

            % replace first pass constant by vector of ones
            if params.addconstant == 1
                phi(:,1) = 1;
            end

            for i = 1:size(X,2)
                F(:,i,ii) =(phi(~isnan(X(:,i)),:)'*phi(~isnan(X(:,i)),:))\(phi(~isnan(X(:,i)),:)'*X(~isnan(X(:,i)),i));
                [warnmsg, msgid] = lastwarn;
                if strcmp(msgid,'MATLAB:singularMatrix')
                    F(:,i,ii) = (phi(~isnan(X(:,i)),:)'*phi(~isnan(X(:,i)),:)+eps*eye(size(phi,2)))\(phi(~isnan(X(:,i)),:)'*X(~isnan(X(:,i)),i));
                end
            end

            % compute the factor estimates
            %F = (phi'*phi)\(phi'*X);

            % singular value catch
            %[warnmsg, msgid] = lastwarn;
            %if strcmp(msgid,'MATLAB:singularMatrix')
            %    F = (phi'*phi+eye(size(phi,2)))\(phi'*X);
            %end


        end
        
        if params.addconstant == 1
           F = F(2:end,:,:);
        end
        
        % store factor estimates
        results.F = F;
        
    elseif params.FP_type == "VBQRTVP"   
        
        % extract dtm
        X = params.dtm';
         
        if max(size(params.quant))>1
            F = zeros(size(results.phi_hat,2),size(X,2),max(size(params.quant))); 
        else
            F = zeros(size(results.phi_hat,2),size(X,2)); 
        end
        
        
        
        for ii = 1:max(size(params.quant))
            % extract the coefficients
            phi = results.phi_hat(:,:,:,ii);

            % replace first pass constant by vector of ones
            if params.addconstant == 1
                phi(:,1,:) = 1;
            end

            for i = 1:size(X,2)
                if params.addconstant==0
                    xx = squeeze(phi(i,:,~isnan(X(:,i)),:));
                else
                    xx = squeeze(phi(i,:,~isnan(X(:,i)),:))';
                end
                F(:,i,ii) =(xx'*xx)\(xx'*X(~isnan(X(:,i)),i));
                [warnmsg, msgid] = lastwarn;
                if strcmp(msgid,'MATLAB:singularMatrix')
                    F(:,i,ii) = (xx'*xx+eps*eye(size(xx,2)))\(xx*X(~isnan(X(:,i)),i));
                end
            end

            % compute the factor estimates
            %F = (phi'*phi)\(phi'*X);

            % singular value catch
            %[warnmsg, msgid] = lastwarn;
            %if strcmp(msgid,'MATLAB:singularMatrix')
            %    F = (phi'*phi+eye(size(phi,2)))\(phi'*X);
            %end


        end
        
        if params.addconstant == 1
           F = F(2:end,:,:);
        end
        
        % store factor estimates
        results.F = F;
       
    elseif params.FP_type == 'QRFactors'
         
        % extract the coefficients
        phi = results.phi_hat';
        
        % replace first pass constant by vector of ones
        if params.addconstant == 1
            phi = [ones(size(phi,1),1), phi];
        end
        
        % extract dtm
        X = params.dtm';
        
        F = zeros(size(phi,2),size(X,2)); 
        for i = 1:size(X,2)
            F(:,i) =(phi(~isnan(X(:,i)),:)'*phi(~isnan(X(:,i)),:))\(phi(~isnan(X(:,i)),:)'*X(~isnan(X(:,i)),i));
            [warnmsg, msgid] = lastwarn;
            if strcmp(msgid,'MATLAB:singularMatrix')
                F(:,i) = (phi(~isnan(X(:,i)),:)'*phi(~isnan(X(:,i)),:)+eps*eye(size(phi,2)))\(phi(~isnan(X(:,i)),:)'*X(~isnan(X(:,i)),i));
            end
        end
        
        if params.addconstant == 1
            F = F(2:end,:);
        end

        % store factor estimates
        results.F = F;
        
    elseif params.FP_type == "OLS" || params.FP_type == "OLS restricted" ||  params.FP_type == "Hard Thresholding"  
        % extract the coefficients
        phi = results.phi_hat';
        
        % replace first pass constant by vector of ones
        if params.addconstant == 1
            phi(:,1) = 1;
        end
        
        % extract dtm
        X = params.dtm';
        
        F = zeros(size(phi,2),size(X,2)); 
        for i = 1:size(X,2)
            F(:,i) =(phi(~isnan(X(:,i)),:)'*phi(~isnan(X(:,i)),:))\(phi(~isnan(X(:,i)),:)'*X(~isnan(X(:,i)),i));
            [warnmsg, msgid] = lastwarn;
            if strcmp(msgid,'MATLAB:singularMatrix')
                F(:,i) = (phi(~isnan(X(:,i)),:)'*phi(~isnan(X(:,i)),:)+eps*eye(size(phi,2)))\(phi(~isnan(X(:,i)),:)'*X(~isnan(X(:,i)),i));
            end
        end

        % compute the factor estimates
        %F = (phi'*phi)\(phi'*X);
        
        % singular value catch
        %[warnmsg, msgid] = lastwarn;
        %if strcmp(msgid,'MATLAB:singularMatrix')
        %    F = (phi'*phi+eye(size(phi,2)))\(phi'*X);
        %end
        
        if params.addconstant == 1
            F = F(2:end,:);
        end

        % store factor estimates
        results.F = F;
        
    elseif params.FP_type == "Forgetting Factor" %|| params.FP_type == "TVP Lasso"
        
        % read out stored coefficients
        phi = results.phi_hat;
        
        % replace first pass constant by vector of ones
        if params.addconstant == 1
            phi(:,1,:) = 1;
        end
        
        % extract dtm
        X = params.dtm';
        
        % initalize factor
        F = zeros(params.k+params.addconstant, params.T);
        
        % iterate through time
        for i = 1:1:params.T
            phit = squeeze(phi(i,:,:))';
            Xt = X(:,i);
            
            % compute factor estimates for all t
            F(:,i) = (phit'*phit)\(phit'*Xt);  
        end

        if params.addconstant == 1
            % drop constant
            F = F(2:end,:);
        end
        
        % store factor estimates
        results.F = F;
        
    elseif params.FP_type == "TVP Lasso" || params.FP_type == "Variational Bayes"
        
        % read out stored coefficients
        phi = results.phi_hat;
       
        % replace first pass constant by vector of ones
        if params.addconstant == 1
            phi(:,1,:) = 1;
        end
        
        % extract dtm
        X = params.dtm';
        
        % initalize factor
        F = zeros(params.k+params.addconstant, params.T);
        
        % iterate through time
        for i = 1:1:params.T
            if size(phi, 2) == 1
                phit = squeeze(phi(i,:,:));
            else
                phit = squeeze(phi(i,:,:))';
            end
            nzid = sum(phit(:,params.addconstant+1:end),2)~=0;
            
            Xt = X(:,i);
            
            % compute factor estimates for all t
            F(:,i) = (phit(nzid,:)'*phit(nzid,:))\(phit(nzid,:)'*Xt(nzid,:));  
        end

        if params.addconstant == 1
            % drop constant
            F = F(2:end,:);
        end
        
        % store factor estimates
        results.F = F;
        
    end
end

