function p = hypergeom_cdf(x,m,k,n)
% hypergeom_cdf - Hypergeometric cumulative density function
%
% Synopsis:
%   y = hypergeom_cdf(x,m,k,n)
%   
% Arguments:
%  x: Evaluate the cdf at x. Representing the number of white balls drawn
%     without replacement from an urn which contains both black and white
%     balls.
%  m: Integer parameter m. The total number of balls in the urn
%  k: Integer parameter k. The number of white balls in the urn.
%  n: Integer parameter n, The number of balls drawn from the urn.
%   
% Returns:
%  y: [a b] matrix. Value of the cumulative density function
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
% See also: hypergeom_pdf
% 

% Author(s), Copyright: Anton Schwaighofer, Oct 2005, based on code from
% the statistics toolbox
% $Id: hypergeom_cdf.m,v 1.1 2005/10/07 09:33:16 neuro_toolbox Exp $

error(nargchk(4, 4, nargin));

%Initialize P to zero.
p = zeros(size(x));

outlim = ( x<0 | m<0 | k<0 | n<0 | round(m)~=m | round(k)~=k | round(n)~=n | n>m ...
           | k>m | x>n | x>k);
if any(outlim),
  p(outlim) = NaN;
end
k1 = find(outlim);

kc = (1:prod(size(x)))';
kc(k1) = [];

% Compute p when xx >= 0.
if any(kc)
  xx = floor(x);
  val = min([max(max(k(kc))) max(max(xx(kc))) max(max(n(kc)))]);
  i1 = [0:val]';
  compare = i1(:,ones(size(kc)));
  index = xx(kc);
  index = index(:);
  index = index(:,ones(size(i1)))';
  mbig = m(kc);
  mbig = mbig(:);
  mbig = mbig(:,ones(size(i1)))';
  kbig = k(kc);
  kbig = kbig(:);
  kbig = kbig(:,ones(size(i1)))';
  nbig = n(kc);
  nbig = nbig(:);
  nbig = nbig(:,ones(size(i1)))';
  p0 = hygepdf(compare,mbig,kbig,nbig);
  indicator = find(compare > index);
  p0(indicator) = zeros(size(indicator));
  p(kc) = sum(p0);
end

% Make sure that round-off errors never make P greater than 1.
k1 = find(p > 1);
if ~isempty(k1),
  p(k1) = 1;
end
