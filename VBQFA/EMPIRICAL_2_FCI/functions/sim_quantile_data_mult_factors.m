function [Y, fit, y_, F] = sim_quantile_data_mult_factors(T,N,ps)
    
    % matrix of simulated series 
    Y = zeros(T,N);
    y_ = zeros(T,length(ps),N);
    
    % for now simulate one factor following an AR(1)
    k = length(ps);
    
    F = zeros(T+1,k);
    for i = 2:T+1
        F(i,:) = 0.9*F(i-1,:) + 0.1*randn(1,k); % 0.9
    end
    F = F(2:end,:);
    a = 0;
    b = 1;
    
    
    for n = 1:N
        lam = a+(b-a).*rand(1,length(ps));

        [~,id] = sort(abs(lam));
        lam = abs(lam(id));

        y = F.*fliplr(lam);
        x = icdf('Normal',ps,0,1);
        y = y + x;

        ResMatch = Step2match_mod(y, [], ps);

        for i = 1:T
            Y(i,n) = rskt(1,ResMatch.STpar(i,1),ResMatch.STpar(i,2),ResMatch.STpar(i,3),ResMatch.STpar(i,4));
        end
        
        y_(:,:,n) = y;  
        fit(n)    = ResMatch;
    end
end