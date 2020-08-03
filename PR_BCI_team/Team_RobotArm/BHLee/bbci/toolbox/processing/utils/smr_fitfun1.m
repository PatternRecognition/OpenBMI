function err = smr_fitfun1(lambda,t,y)
%   SMR_FITFUN1(lambda,t,y) returns the error between the data and the values
%   computed by the current function of lambda.
%
%   SMR_FITFUN1 assumes a function of the form
%
%     y = c(1) + c(2) / t^lambda 
%
%   with linear parameters c(i) and nonlinear parameter lambda.

A = ones(length(t),2);
A(:,2) = 1 ./ (t.^lambda);
c = A\y;
z = A*c;
err = norm(z-y);