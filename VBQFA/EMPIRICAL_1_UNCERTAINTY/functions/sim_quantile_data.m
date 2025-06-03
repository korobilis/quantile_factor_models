function [Y, ResMatch, y, F] = sim_quantile_data(n,ps,d)
    
    % for now simulate one factor following an AR(1)
    F = zeros(n+1,1);
    for i = 2:n+1
        F(i,:) = 0.9*F(i-1,:) + 0.25*randn(1); % 0.9
    end
    F = F(2:end,:);
    a = 0;
    b = 1;
    lam = a+(b-a).*rand(1,length(ps));
    
    [~,id] = sort(abs(lam));
    lam = abs(lam(id));
    
    y = F.*fliplr(lam);
    x = icdf('Normal',ps,0,1);
    y = y + x;
    
    ResMatch = Step2match_mod(y, [], ps);
    
    
    Y = zeros(n,d);
    for i = 1:n
        Y(i,:) = rskt(d,ResMatch.STpar(i,1),ResMatch.STpar(i,2),ResMatch.STpar(i,3),ResMatch.STpar(i,4));
    end
end