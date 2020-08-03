function B = inplacecolumn(A, k)
% function B = inplacecolumn(A, k)
%
% Return the inplace-column A(:,k)
%
% Important notes:
%
% - use MEX function releaseinplace(B) to release properly shared-data
%   pointer before clear/reuse B.
% - All inplace variables shared data with A must be released before
%   the original array A is cleared/reused.
%
% See also: inplacearray, releaseinplace
% 
% Author Bruno Luong <brunoluong@yahoo.com>
% Last update: 27/June/2009

[M N] = size(A); %#ok trick m-lint
B = inplacearray(A, (k-1)*M, [M 1]);