% function releaseinplace(b)
%
% Release the data from an inplace column mxArray that was created
% with the inplacecolumn function.
%
% - Must be used before the inplace variable is cleared/reused
%   and before the original variable is cleared
% - For advanced users only!!!! In any case use at your own risk
%
% See also: inplacearray, inplacecolumn
% 
% Author Bruno Luong <brunoluong@yahoo.com>
% Last update: 27/June/2009