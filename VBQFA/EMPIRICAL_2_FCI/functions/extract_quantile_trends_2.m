function [trend] = extract_quantile_trends_2(x,quant)

    [T,p] = size(x);
    nq = length(quant);     
    
    trend = zeros(T,p,nq);
    for j = 1:p
        [trend(:,j,:)] = VBQLL(x(:,j),100,quant);                  
    end
end

