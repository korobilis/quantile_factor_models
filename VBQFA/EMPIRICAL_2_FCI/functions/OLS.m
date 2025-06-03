function [Yfore] = OLS(Y,X,Xfore,nsave)

beta = (X'*X)\(X'*Y);
sigma = (Y-X*beta)'*(Y-X*beta)/(size(Y,1)-size(X,2));

Yfore = Xfore*beta + sqrt(sigma)*randn(nsave,1);