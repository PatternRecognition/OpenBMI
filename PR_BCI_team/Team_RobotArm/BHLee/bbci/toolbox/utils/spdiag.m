function A = spdiag(d)
% spdiag - Generate a sparse diagonal matrix
%
% Synopsis:
%   A = spdiag(d)
%   
% Arguments:
%  d: [1 N] or [N 1] vector. Elements to put along the diagonal.
%   
% Returns:
%  A: [N N] sparse matrix with element A(i,i)==d(i)
%   
% Description:
%   Similar to diag, spdiag creates a sparse diagonal matrix, avoiding
%   the cryptic syntax of spdiags.
%   
%   
% Examples:
%   spdiag([1 2])
%     returns
%      (1,1)        1
%      (2,2)        2
%
%   
% See also: spdiags
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2005
% $Id: spdiag.m,v 1.1 2005/06/02 20:03:01 neuro_toolbox Exp $


error(nargchk(1, 1, nargin));

if min(size(d))>1,
  error('Input argument d must be a vector');
end
N = length(d);
A = spdiags(d(:), 0, N, N);
