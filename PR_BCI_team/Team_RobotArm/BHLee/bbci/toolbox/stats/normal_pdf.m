function [y,logy] = normal_pdf(x,mu,sigma)
% normal_pdf - Normal probability density function (pdf)
%
% Synopsis:
%   y = normal_pdf(x)
%   [y,logy] = normal_pdf(x,mu,sigma)
%   
% Arguments:
%  x: [m n] matrix. Argument for the normal pdf
%  mu: [m n] matrix. Mean value of the normal distribution. Default value: 0
%  sigma: [m n] matrix. Standard deviation of the normal distribution. Default
%      value: 1
%   
% Returns:
%  y: Scalar, value of the normal pdf
%  logy: Scalar, equal to log(y).
%   
% Description:
%   The value of the "Gaussian bump" at argument x. If non-scalar arguments
%   are given, the pdf is computed for each entry (i.e., the multi-variate
%   Gaussian is not supported).  All computations are done on log scale,
%   output argument logy can thus be expected to be more accurate for
%   small values of the normal pdf.
%   
%   
% Examples:
%   normal_pdf(0)
%     returns 0.39894, the value of the pdf for the standard normal
%     distribution at 0
%   figure;fplot('normal_pdf', [-2 10], [],[],[],4, 2)
%     plots the Gaussian PDF with mean 4 and standard deviation 2.
%   
% See also: normal_cdf,normal_invcdf
% 

% Author(s), Copyright: Anton Schwaighofer, April 2006
% $Id: normal_pdf.m,v 1.2 2006/06/19 19:46:53 neuro_toolbox Exp $

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

logy = -0.5*log(2*pi)-log(sigma)-0.5.*(((x-mu)./sigma).^2);
y = exp(logy);
