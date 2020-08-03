function p = t_cdf(x,v)
% t_cdf - Cumulative distribution function (cdf) of Student's T distribution
%
% Synopsis:
%   p = t_cdf(x,v)
%   
% Arguments:
%  x: [n m] matrix. The cdf is computed at the values x given here
%  v: [n m] matrix. Degrees of freedom parameter for student
%   
% Returns:
%  p: [n m] matrix. Values of the cdf
%   
% Description:
%   Scalar inputs for either x or v are expanded to constant matrices the
%   size of the other input.
%   
%   
% Examples:
%   t_cdf(1.2, [2 3])
%     computes that CDF at 1.2 for t-distributions with either 2 or 3
%     degrees of freedom.
%
% References:
%   [1] M. Abramowitz and I. A. Stegun, "Handbook of Mathematical
%   Functions", Government Printing Office, 1964, 26.7.
%   [2] L. Devroye, "Non-Uniform Random Variate Generation",
%   Springer-Verlag, 1986
%   [3] E. Kreyszig, "Introductory Mathematical Statistics",
%   John Wiley, 1970, Section 10.3, pages 144-146.
% 
% See also: paired_ttest
% 

% Author(s), Copyright: Anton Schwaighofer, Sep 2005
% $Id: t_cdf.m,v 1.1 2005/09/02 15:13:55 neuro_toolbox Exp $

normcutoff = 1e7;
error(nargchk(2, 2, nargin));

% Initialize P to zero.
p=zeros(size(x));

if prod(size(v))==1,
  v = repmat(v, size(x));
elseif prod(size(x))==1,
  x = repmat(x, size(v));
end

% use special cases for some specific values of v
k = find(v==1);
% See Devroye pages 29 and 450.  (This is also the Cauchy distribution)
if any(k)
  p(k) = .5 + atan(x(k))/pi;
end
k = find(v>=normcutoff);
if any(k)
  p(k) = normal_cdf(x(k));
end

% See Abramowitz and Stegun, formulas 26.5.27 and 26.7.1
k = find(x ~= 0 & v ~= 1 & v > 0 & v < normcutoff);
if any(k),                            % first compute F(-|x|)
  xx = v(k) ./ (v(k) + x(k).^2);
  p(k) = betainc(xx, v(k)/2, 0.5)/2;
end

% Adjust for x>0.  Right now p<0.5, so this is numerically safe.
k = find(x > 0 & v ~= 1 & v > 0 & v < normcutoff);
if any(k), p(k) = 1 - p(k); end

p(x == 0 & v ~= 1 & v > 0) = 0.5;

% Return NaN for invalid inputs.
p(v <= 0 | isnan(x) | isnan(v)) = NaN;

