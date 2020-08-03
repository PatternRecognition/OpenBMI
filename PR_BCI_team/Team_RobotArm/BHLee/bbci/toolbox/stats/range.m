function y = range(x,dim)
% range - The range is the difference between maximum and minimum values
%
% Synopsis:
%   y = range(x)
%   y = range(x,dim)
%   
% Arguments:
%   x: A matrix or vector
%   dim: Dimension along which min and max operate. Default: first
%        non-singleton dimension
%   
% Returns:
%   y: Difference between min and max values along the chosen dimension
%   
%   
% See also: min,max
% 

% Author(s): Anton Schwaighofer, Jan 2005
% $Id: range.m,v 1.2 2006/02/22 15:49:58 neuro_toolbox Exp $

error(nargchk(1, 2, nargin));
if nargin<2,
  % Operate along the first non-singleton dimension
  dim = min(find(size(x)~=1));
  if isempty(dim), 
    dim = 1; 
  end
end

xmin = min(x, [], dim);
xmax = max(x, [], dim);
y = xmax-xmin;
