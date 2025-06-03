function [forecasts, ref] = f_back_transform(params, results, Y)
    
    % find starting point of forecasting exercise
    id_ = find(params.forestart == params.dates);

    % read out original data for back transformation
    datapoint = table2array(Y(params.dates,:));
   
    ref       = NaN(size(results.ref));
    forecasts = NaN(size(results.ref));
    for i = size(results.ref,3)-1:-1:1
        ref(:,:,i)       = exp(cumsum(results.ref(:,:,i))+log(datapoint(id_+i-1,:)));
        forecasts(:,:,i) = exp(cumsum(results.forecasts(:,:,i))+log(datapoint(id_+i-1,:)));
    end    

end

