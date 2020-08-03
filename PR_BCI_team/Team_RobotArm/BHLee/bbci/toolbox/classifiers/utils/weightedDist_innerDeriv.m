function L = weightedDist_innerDeriv(X1, X2, dim)
% weightedDist_innerDeriv - Derivate of weighted distance term wrt. input weights
%
% Synopsis:
%   L = weightedDist_innerDeriv(X1,X2,dim)
%   
% Arguments:
%   X1: [d N1] matrix, point set 1
%   X2: [d N2] matrix, point set 2. If left empty, X2==X1 is assumed
%   dim: Integer, range 1...d. Choose the derivative
%   
% Returns:
%   L: [N1 N2] matrix, the derivative
%   
% Description:
%   This helper function is useful for all kernel derivative
%   computations, where the innermost part of the kernel is a weighted
%   distance term.
%   
% See also: weightedDist,kernderiv_rbf
% 

% Author(s), Copyright: Anton Schwaighofer, Mar 2005
% $Id: weightedDist_innerDeriv.m,v 1.5 2006/06/19 19:59:14 neuro_toolbox Exp $


error(nargchk(3, 3, nargin));

N1 = size(X1, 2);
X1i = X1(dim,:);
if isempty(X2),
  L = (X1i.*X1i)'*ones([1 N1]);
  L = L+L';
  L = L - (2*X1i')*X1i;
else
  N2 = size(X2, 2);
  X2i = X2(dim,:);
  L = (X1i.*X1i)'*ones([1 N2]) - (2*X1i')*X2i + ones([N1 1])*(X2i.*X2i);
end
