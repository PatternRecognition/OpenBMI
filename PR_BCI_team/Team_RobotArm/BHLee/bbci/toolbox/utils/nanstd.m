function y = nanstd(x, flag, dim)
% nanstd - Standard deviation ignoring NaNs
%
% Synopsis:
%   y = nanstd(x)
%   y = nanstd(x,flag)
%   y = nanstd(x,flag,dim)
%   
% Arguments:
%   x: Matrix or vector
%   flag: Logical. Default: false (normalize by N-1). If true: normalize by N
%       (produce the second moment about the mean)
%   dim: Dimension along which to operate
%   
% Returns:
%   y: Standard deviation along the chosen dimension
%   
% Description:
%   nanstd(X) returns the standard deviation treating NaNs 
%   as missing values.
%   For vectors, nanstd(X) is the standard deviation of the
%   non-NaN elements in X.  For matrices, nanstd(X) is a row
%   vector containing the standard deviation of each column,
%   ignoring NaNs.
%   
% Examples:
%   nanstd([1 2 NaN]) returns 0.7071 (the same result as std([1 2])
%   nanstd([1 2 NaN; 3 4 5], 1, 1) returns [1 1 0]
%   
% See also: nanmean,nansum
% 

% Author(s): Anton Schwaighofer, Feb 2005
% $Id: nanstd.m,v 1.1 2005/02/15 11:55:59 neuro_toolbox Exp $

if nargin<3,
  % Operate along the first non-singleton dimension
  dim = min(find(size(x)~=1));
  if isempty(dim), 
    dim = 1; 
  end
end
if nargin<2,
  flag = 0;
end
if isempty(x),
  y = 0/0;
  return;
end
tile = ones(1, max(ndims(x), dim));
tile(dim) = size(x,dim);
% Effective number of non-NaNs
effSize = size(x,dim)-sum(isnan(x),dim);
% Remove mean, also ignoring NaNs
xc = x - repmat(nanmean(x, dim), tile);
if flag,
  y = sqrt(nansum(conj(xc).*xc, dim)./effSize);
else
  y = sqrt(nansum(conj(xc).*xc, dim)./(effSize-1));
  % Simulate the behaviour of std: if only one sample, output 0
  y(effSize==1) = 0;
end
