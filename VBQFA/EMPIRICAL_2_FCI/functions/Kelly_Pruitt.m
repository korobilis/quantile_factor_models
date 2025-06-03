function [QFKP] = Kelly_Pruitt(y,x,k,quant,h,method)
    
    %% Preliminaries
    X = [ones(length(y)-h,1) x(1:end-h,:)];
    Y  = y(1+h:end,:);
   
    phi = zeros(length(Y),size(X,2),length(quant));
    QFKP = zeros(size(x,1),k,length(quant));
    
    if isempty(method)
        method = 'Original';
    end
    
    switch method
        case 'Original'
            for q = 1:length(quant)
                %% |=== First Pass
                for i = 2:size(X,2)
                    phi(end,[1,i],q) = rq_fnm(X(:,[1,i]), Y, quant(q));
                end 
                
                %% |=== Second Pass
                QFKP(:,:,q) = x(:,:)/squeeze(phi(end,2:end,q));
            end
        case 'VB'
            for q = 1:length(quant)
                %% |=== First Pass
                [phi(:,:,q),~,~,~] = VBTVPQR(Y,X,500,quant(q),0,10);
                               
                %% |=== Second Pass
                QFKP(:,:,q) = x(:,:)/squeeze(phi(end,2:end,q));
            end
    end
    
end

