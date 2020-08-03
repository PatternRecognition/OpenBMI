function D = weightedInnerProd(X1,X2,inweights)
% weightedInnerProd - Weighted inner product between point sets
%
% Synopsis:
%   D = weightedInnerProd(X1,X2,inweights)
%   
% Arguments:
%   X1: [d N1] matrix. Input data 1 for distance computation, N1 points in d
%       dimensions
%   X2: [d N2] matrix. Input data 2, N2 points in d dimensions
%   inweights: [1 1], [1 d] or [d d] matrix. Input weights, uniform for all
%       dimensions, per dimension, or a full transformation matrix
%   
% Returns:
%   D: [N1 N2] matrix, weighter inner products between points
%       X1(:,i) and X2(:,j) in element D(i,j)
%   
% Description:
%   This function computes weighted inner products, with a specific
%   scaling/transformation applied to each dimension before distance
%   computation.
%   The scaling parameter 'inweights' expected here is in fact the
%   *squared* scaling applied to each dimension, or, in the case of a
%   matrix scaling, R*R', if R is the original transformation matrix.
%
%   For vectors X1, X2 of length d, the result returned is
%     X1'*inweights*X2
%   
% Examples:
%   weightedInnerProd(1, 2, 1)
%     returns 2
%   weightedInnerProd(1, 2, 0.5*0.5)
%     returns the inner prodcut after scaling each dimension by a factor of
%     0.5, which gives 0.5.
%   weightedInnerProd([1;1], [1;2], [1 0.5].^2)
%     returns 1.5 (scaling dimension 1 by 1, dimension 2 by 0.5)
%   
% 

% Author(s), Copyright: Joaquin Quiñonero Candela, Mar 2006
% $Id: weightedInnerProd.m,v 1.1 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(1, 3, nargin));

if nargin<3,
  inweights = 1;
end
if nargin<2,
  X2 = [];
end
[ndims N1] = size(X1);
if ~isempty(X2),
  if ndims~=size(X2,1),
    error('Number of dimensions in X1 and X2 must match');
  end
  N2 = size(X2,2);
end

if prod(size(inweights))==1,
  % Scalar inweights
  beta = inweights;
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
if isempty(X2),
  X2 = X1;
end
D = X1' * beta * X2;
