function [x] = Normal(a,b,n,p)
% Note that a is the mean, and b is the variance
x = a + sqrt(b)*randn(n,p);
end