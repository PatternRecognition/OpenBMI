function y = nanmean(x,dim)
% nanmean - Compute mean ignoring NaNs.
%
% Synopsis:
%   y = nanmean(x,dim)
%   
% Arguments:
%   x: Matrix or vector
%   dim: Dimension along which sum operates. Default: First non-singleton
%       dimension
%   
% Returns:
%   y: Mean along the chosen dimension, treating NaNs as missing values.
%   
% Description:
%   nanmean(X) returns the average treating NaNs as missing values.
%   For vectors, nanmean(X) is the mean value of the non-NaN
%   elements in X.  For matrices, nanmean(X) is a row vector
%   containing the mean value of each column, ignoring NaNs.
%
%   nanmean(X,dim) operates along the dimension dim. 
%   
% Examples:
%   nanmean([1 2 NaN]) returns 1.5
%   nanmean([1 2 NaN],1) returns [1 2 NaN]
%   
% See also: nansum,nanstd
% 

% Author(s): Anton Schwaighofer, Feb 2005
% $Id: nanmean.m,v 1.1 2005/02/15 11:55:59 neuro_toolbox Exp $

if nargin<2,
  % Operate along the first non-singleton dimension
  dim = min(find(size(x)~=1));
  if isempty(dim), 
    dim = 1; 
  end
end
effSize = size(x,dim)-sum(isnan(x),dim);
% Avoid Matlab division by zero warning for summing over all NaNs (NaN/0)
effSize(effSize==0) = 1;
y = nansum(x,dim)./effSize;
