function y = hypergeom_pdf(x,m,k,n)
% hypergeom_pdf - Hypergeometric probability density function
%
% Synopsis:
%   y = hypergeom_pdf(x,m,k,n)
%   
% Arguments:
%  x: Evaluate the pdf at x. Representing the number of white balls drawn
%     without replacement from an urn which contains both black and white
%     balls.
%  m: Integer parameter m. The total number of balls in the urn
%  k: Integer parameter k. The number of white balls in the urn.
%  n: Integer parameter n, The number of balls drawn from the urn.
%   
% Returns:
%  y: [a b] matrix. Value of the density function
%   
% Description:
%   Note: The density function is zero unless x is an integer.
%   
%   
% Examples:
%   
%   
% References:
%   Mood, Alexander M., Graybill, Franklin A. and Boes, Duane C.,
%   "Introduction to the Theory of Statistics, Third Edition", McGraw
%   Hill, 1974 p. 91.
%   
% See also: hypergeom_cdf,gammaln
% 

% Author(s), Copyright: Anton Schwaighofer, Oct 2005, based on code from
% the statistics toolbox
% $Id: hypergeom_pdf.m,v 1.1 2005/10/07 09:33:16 neuro_toolbox Exp $

error(nargchk(4, 4, nargin));

y = zeros(size(x));

%   Return NaN for values of the parameters outside their respective
%   limits.
outlim = ( m<0 | k<0 | n<0 | round(m)~=m | round(k)~=k | round(n)~=n | n>m ...
           | k>m | x>n | x>k);
if any(outlim),
  y(outlim) = NaN;
end
k1 = find(outlim);

kc = 1:prod(size(x));

%   Remove values of X for which Y is zero by inspection.
yzero = x(kc)>k(kc) | m(kc)-k(kc)-n(kc)+x(kc)+1<=0 | x(kc) < 0;
k2 = find(yzero);

k12 = vertcat(k1(:), k2(:));
kc(k12) = [];

% find integer values of x that are within the correct range
if ~isempty(kc),
  kx = gammaln(k(kc)+1)-gammaln(x(kc)+1)-gammaln(k(kc)-x(kc)+1);
  mn = gammaln(m(kc)+1)-gammaln(n(kc)+1)-gammaln(m(kc)-n(kc)+1);
  mknx = gammaln(m(kc)-k(kc)+1)-gammaln(n(kc)-x(kc)+1)-gammaln(m(kc)-k(kc)-(n(kc)-x(kc))+1);                      
  y(kc) = exp(kx + mknx - mn);
end
