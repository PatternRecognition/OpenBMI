function y = nanmax(x,dim)
% nanmax - Max ignoring NaNs.
%
% Synopsis:
%   y = nanmax(x)
%   y = nanmax(x,dim)
%   
% Arguments:
%   x: Matrix or vector
%   dim: Dimension along which sum operates. Default: First non-singleton
%       dimension.
%   
% Returns:
%   y: max along the chosen dimension, treating NaNs as missing values.
%   
% Description:
%   For vectors, nanmax(X) is the max of the non-NaN elements in
%   X. For matrices, nanmax(X) is a row vector containing the max
%   of the non-NaN elements in each column of X. For N-D arrays,
%   nanmax(X) operates along the first non-singleton dimension.
%
%   nanmax(X,dim) determaxes the max along the dimension dim. 
%   
% Examples:
%   nanmax([1 2 NaN]) returns 2.
%   nanmax([1 2 NaN], 1) returns [1 2 NaN].
%   nanmax([1 2 NaN], 2) returns 2.
%   
% See also: nansum,nanmean,nanstd
% 

% Author(s): Benjamax Blankertz, Anton Schwaighofer, Aug 2005

if nargin<2,
  % Operate along the first non-singleton dimension
  dim = max(find(size(x)~=1));
  if isempty(dim),
    dim = 1; 
  end
end
% Replace NaNs with zeros.
nans = isnan(x);
x(nans) = -inf;

% Protect against an entire column of NaNs
y = max(x, [], dim);
allNaNs = all(nans, dim);
y(allNaNs) = NaN;
