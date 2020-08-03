function y = nansum(x,dim)
% nansum - Sum ignoring NaNs.
%
% Synopsis:
%   y = nansum(x)
%   y = nansum(x,dim)
%   
% Arguments:
%   x: Matrix or vector
%   dim: Dimension along which sum operates. Default: First non-singleton
%       dimension.
%   
% Returns:
%   y: Sum along the chosen dimension, treating NaNs as missing values.
%   
% Description:
%   For vectors, nansum(X) is the sum of the non-NaN elements in
%   X. For matrices, nansum(X) is a row vector containing the sum 
%   of the non-NaN elements in each column of X. For N-D arrays,
%   nansum(X) operates along the first non-singleton dimension.
%
%   nansum(X,dim) sums along the dimension dim. 
%   
% Examples:
%   nansum([1 2 NaN]) returns 3.
%   nansum([1 2 NaN], 1) returns [1 2 NaN].
%   nansum([1 2 NaN], 2) returns 3.
%   
% See also: nanmean,nanstd
% 

% Author(s): Anton Schwaighofer, Feb 2005
% $Id: nansum.m,v 1.2 2005/05/23 11:34:35 neuro_toolbox Exp $

if nargin<2,
  % Operate along the first non-singleton dimension
  dim = min(find(size(x)~=1));
  if isempty(dim),
    dim = 1; 
  end
end
% Replace NaNs with zeros.
nans = isnan(x);
x(nans) = 0;

% Protect against an entire column of NaNs
y = sum(x, dim);
allNaNs = all(nans, dim);
y(allNaNs) = NaN;
