function D = weightedSquareNorm(X,inweights)
% weightedSquareNorm - Weighted square norm of a set of vectors
%
% Synopsis:
%   D = weightedSquareNorm(X,inweights)
%   
% Arguments:
%   X: [d N] matrix. Input data, N points in d dimensions
%   inweights: [1 1], [1 d] or [d d] matrix. Input weights, uniform for all
%       dimensions, per dimension, or a full transformation matrix. If
%       not given, inweights is taken to be 1.
%   
% Returns:
%   D: [N 1] matrix, weighted square norm of each of the N vectors in X
%   
% Description:
%   This function computes weighted square norms, with a specific
%   scaling/transformation applied to each dimension before distance
%   computation.
%   The scaling parameter 'inweights' expected here is in fact the
%   *squared* scaling applied to each dimension, or, in the case of a
%   matrix scaling, R*R', if R is the original transformation matrix.
%
%   For a vector X of length d, the result returned is
%     X'*inweights*X
%   
% Examples:
%   weightedInnerProd(1, 1)
%     returns 1
%   weightedInnerProd(1, 0.5*0.5)
%     returns the inner prodcut after scaling each dimension by a factor of
%     0.5, which gives 0.25.
%   weightedInnerProd([1;1], [1 0.5].^2)
%     returns 1.25 (scaling dimension 1 by 1, dimension 2 by 0.5)
%   
% 

% Author(s), Copyright: Joaquin Quiñonero Candela, Mar 2006
% $Id: weightedSquareNorm.m,v 1.1 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(1, 2, nargin));

if nargin<2,
  inweights = 1;
end

[ndims N] = size(X);

if prod(size(inweights))==1,
  % Scalar inweights
  beta = inweights;
  D = beta * sum(X'.^2,2);
  return
elseif all(size(inweights)==[1 ndims]) | all(size(inweights)==[ndims 1]),
  % Vector inweights: assume dimension-wise scaling
  % Scaling with sparse diagonal matrix is much faster
  beta = speye(ndims);
  % Put weights along the diagonal
  beta(1:ndims+1:ndims*ndims) = inweights;
elseif size(inweights)==[ndims ndims],
  beta = inweights;
else
  error('Length of ''inweights'' parameter must match input dimensions');
end

% Compute the weighted inner product between X1 and X2

D = sum((X' * beta).* X',2);
