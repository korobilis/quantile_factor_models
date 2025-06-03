function [Q,tauh,delta,psijh] = Bhattacharya_Dunson_prior(beta,tauh,delta,df,ad1,bd1,ad2,bd2,p,k)

%   |------------------ 
%   df  = gamma hyperparameters for t_{ij}
%   ad1, bd1 = gamma hyperparameters for delta_1
%   ad2, bd2 = gamma hyperparameters delta_h, h >= 2

    %------Update psi_{jh}'s------%
    psijh = gamrnd(df/2 + 0.5, 1./(df/2 + bsxfun(@times,beta.^2,tauh')));
    
    %------Update delta & tauh------%
    mat = bsxfun(@times,psijh,beta.^2);
    ad = ad1 + 0.5*p*k; 
    bd = bd1 + 0.5*(1/delta(1))*sum(tauh.*sum(mat)');
    delta(1) = gamrnd(ad,1/bd);
    tauh = cumprod(delta);

    for h = 2:k
    	ad = ad2 + 0.5*p*(k-h+1); 
        bd = bd2 + 0.5*(1/delta(h))*sum(tauh(h:end).*sum(mat(:,h:end))');
    	delta(h) = gamrnd(ad,1/bd); 
        tauh = cumprod(delta);
    end
    
    %---update precision parameters----%
    Q = 1./bsxfun(@times,psijh,tauh');    

end

