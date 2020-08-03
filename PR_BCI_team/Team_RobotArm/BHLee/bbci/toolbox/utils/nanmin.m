function y = nanmin(x,dim)
% nanmin - Min ignoring NaNs.
%
% Synopsis:
%   y = nanmin(x)
%   y = nanmin(x,dim)
%   
% Arguments:
%   x: Matrix or vector
%   dim: Dimension along which sum operates. Default: First non-singleton
%       dimension.
%   
% Returns:
%   y: min along the chosen dimension, treating NaNs as missing values.
%   
% Description:
%   For vectors, nanmin(X) is the min of the non-NaN elements in
%   X. For matrices, nanmin(X) is a row vector containing the min
%   of the non-NaN elements in each column of X. For N-D arrays,
%   nanmin(X) operates along the first non-singleton dimension.
%
%   nanmin(X,dim) determines the min along the dimension dim. 
%   
% Examples:
%   nanmin([1 2 NaN]) returns 1.
%   nanmin([1 2 NaN], 1) returns [1 2 NaN].
%   nanmin([1 2 NaN], 2) returns 1.
%   
% See also: nansum,nanmean,nanstd
% 

% Author(s): Benjamin Blankertz, Anton Schwaighofer, Aug 2005

if nargin<2,
  % Operate along the first non-singleton dimension
  dim = min(find(size(x)~=1));
  if isempty(dim),
    dim = 1; 
  end
end
% Replace NaNs with zeros.
nans = isnan(x);
x(nans) = inf;

% Protect against an entire column of NaNs
y = min(x, [], dim);
allNaNs = all(nans, dim);
y(allNaNs) = NaN;
