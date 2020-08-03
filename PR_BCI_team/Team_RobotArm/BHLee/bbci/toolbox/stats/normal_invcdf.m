function z = normal_invcdf(p,mu,sigma)
% normal_invcdf - Inverse of the normal cumulative distribution function (cdf)
%
% Synopsis:
%   z = normal_invcdf(p)
%   z = normal_invcdf(p,mu,sigma)
%   
% Arguments:
%  p: [m n] matrix, elements in the range [0,1]
%  mu: [m n] matrix. Mean value of the normal distribution. Default value: 0
%  sigma: [m n] matrix. Standard deviation of the normal
%      distribution. Default value: 1
%   
% Returns:
%  z: Scalar, value of the inverse normal cdf
%   
% Description:
%   This function is the inverse function of normal_cdf, i.e.,
%   normal_invcdf(norma_cdf(x)) returns x. If non-scalar arguments are
%   given, the inverse cdf is computed for each entry
%   
%   
% References:
%   M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
%   Functions", Government Printing Office, 1964, 7.1.1 and 26.2.2
%
% Examples:
%   Make a quantile-quantile plot of residuals after fitting a model:
%     xquant = normal_invcdf(((1:N)-0.5)/N);
%     sres = sort(res/std(res));
%     plot(xquant,sres,'k.')
%   
% See also: normal_cdf,erfcinv
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2005
% $Id: normal_invcdf.m,v 1.2 2005/06/30 17:24:47 neuro_toolbox Exp $

if nargin<3,
  sigma = [];
end
if isempty(sigma),
  sigma = 1;
end
if nargin<2,
  mu = [];
end
if isempty(mu),
  mu = 0;
end

if any(any(p<0 | p>1)),
  error('Argument p must be in the range [0,1]');
end

z = (-sqrt(2)*sigma).*erfcinv(2*p) + mu;
