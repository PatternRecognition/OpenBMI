function err = smr_fitfun2(params, t,y)
%   SMR_FITFUN2(lambda, mu, sigma, t,y) returns the error between the data and the values
%   computed by the current function of lambda.
%
%   SMR_FITFUN2 assumes a function of the form
%
%   y =  c(1) + c(2) / t^lambda + c(3) * phi1(t|mu(1), sigma(1)) + c(4) * phi2(t|mu(2), sigma(2))
%
%   with linear parameters c(i) and nonlinear parameters lambda, mu and sigma.

lambda=params(1);
mu(1) = params(2);
mu(2) = params(3);
sigma(1) = params(4);
sigma(2) = params(5);

A = ones(length(t),4);
A(:,2) = 1 ./ (t.^lambda);
A(:,3) = normpdf(t, mu(1), sigma(1));
A(:,4) = normpdf(t, mu(2), sigma(2));
c = A\y;
z = A*c;
err = norm(z-y);