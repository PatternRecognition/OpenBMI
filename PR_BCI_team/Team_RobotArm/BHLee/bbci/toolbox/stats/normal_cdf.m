function y = normal_cdf(x,mu,sigma)
% normal_cdf - Normal cumulative distribution function (cdf)
%
% Synopsis:
%   y = normal_cdf(x)
%   y = normal_cdf(x,mu,sigma)
%   
% Arguments:
%  x: [m n] matrix. Argument for the normal cdf
%  mu: [m n] matrix. Mean value of the normal distribution. Default value: 0
%  sigma: [m n] matrix. Standard deviation of the normal distribution. Default
%      value: 1
%   
% Returns:
%  y: Scalar, value of the normal cdf (range [0,1])
%   
% Description:
%   The cumulative distribution function is the integral of the "Gaussian
%   bump" from -inf to argument x. If non-scalar arguments are
%   given, the cdf is computed for each entry.
%   
%   
% Examples:
%   normal_cdf(0)
%     returns 0.5, the value of the cdf for the standard normal
%     distribution.
%   normal_cdf(4, 1)
%     returns 0.99865, assuming a normal distribution with mean 1.
%   
% See also: normal_invcdf,erf
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2005
% $Id: normal_cdf.m,v 1.1 2005/06/30 17:20:47 neuro_toolbox Exp $

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

y = 0.5*erf((x-mu)./(sqrt(2).*sigma)) + 0.5;
