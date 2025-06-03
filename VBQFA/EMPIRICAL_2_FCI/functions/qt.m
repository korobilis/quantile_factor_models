function [q] = qt(i,quant)

T = 100000;  probs = rand(T,1);
if i == 1
    error = Normal(0,1^2,T,1);
elseif i == 2
    error = (probs<=1/5).*Normal(-22/25,1^2,T,1) + (probs<=2/5 & probs>1/5).*Normal(-49/125,(3/2)^2,T,1) + (probs>2/5).*Normal(49/250,(5/9)^2,T,1);
elseif i == 3
    error = (probs>1/3).*Normal(0,1^2,T,1) + (probs<=1/3).*Normal(0,(1/10)^2,T,1);
elseif i == 4
    error = (probs<=1/10).*Normal(0,1^2,T,1) + (probs>1/10).*Normal(0,(1/10)^2,T,1);
elseif i == 5
    error = (probs<=1/2).*Normal(-1,(2/3)^2,T,1) + (probs>1/2).*Normal(1,(2/3)^2,T,1);
elseif i == 6
    error = (probs<=1/2).*Normal(-3/2,(1/2)^2,T,1) + (probs>1/2).*Normal(3/2,(1/2)^2,T,1);
elseif i == 7
    error = (probs>=3/4).*Normal(-43/100,1^2,T,1) + (probs<3/4).*Normal(107/100,(1/3)^2,T,1);
else
    error = (probs<=9/20).*Normal(-6/5,(3/5)^2,T,1) + (probs>9/20 & probs<=18/20).*Normal(6/5,(3/5)^2,T,1) + (probs>18/20).*Normal(0,(1/4)^2,T,1);
end

q = quantile(error,quant);
