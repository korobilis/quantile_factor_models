function [c,ceq] = f_nlconstraints_3p(x,beta)

% specify inequality constraints
%c(1) = (L(6,:)*x(13:16))*(L(11,:)*x(13:16));
c(1) = max(x(:,1)*beta(1))-max(x(:,2)*beta(2));
c(2) = max(x(:,2)*beta(2))-max(x(:,3)*beta(3));


% specify equality constraints
%ceq(1)  = x(1:4)'*x(1:4)-1;
%ceq(2)  = x(5:8)'*x(5:8)-1;
%ceq(3)  = x(9:12)'*x(9:12)-1;
%ceq(4)  = x(13:16)'*x(13:16)-1;
%ceq(5)  = x(1:4)'*x(5:8);
%ceq(6)  = x(1:4)'*x(9:12);
%ceq(7)  = x(1:4)'*x(13:16);
%ceq(8)  = x(5:8)'*x(9:12);
%ceq(9)  = x(5:8)'*x(13:16);
%ceq(10) = x(9:12)'*x(13:16);
ceq = [];
end

