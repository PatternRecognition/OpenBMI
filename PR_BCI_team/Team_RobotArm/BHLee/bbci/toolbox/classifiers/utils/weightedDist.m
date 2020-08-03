function D = weightedDist(X1,X2,inweights)
% weightedDist - Weighted Euclidean distance between point sets
%
% Synopsis:
%   D = weightedDist(X1,X2,inweights)
%   
% Arguments:
%   X1: [d N1] matrix. Input data 1 for distance computation, N1 points in d
%       dimensions
%   X2: [d N2] matrix. Input data 2, N2 points in d dimensions
%   inweights: [1 1], [1 d] or [d d] matrix. Input weights, uniform for all
%       dimensions, per dimension, or a full transformation matrix
%   
% Returns:
%   D: [N1 N2] matrix, squared weighted Euclidean distance between points
%       X1(:,i) and X2(:,j) in element D(i,j)
%   
% Description:
%   This function computes squared Euclidean distances, with a specific
%   scaling/transformation applied to each dimension before distance
%   computation.
%   The scaling parameter 'inweights' expected here is in fact the
%   *squared* scaling applied to each dimension, or, in the case of a
%   matrix scaling, R*R', if R is the original transformation matrix.
%
%   For vectors X1, X2 of length d, the result returned is
%     (X1-X2)'*inweights*(X1-X2)
%   
% Examples:
%   weightedDist(0, 2, 1)
%     returns 4 (squared distance)
%   weightedDist(0, 2, 0.5*0.5)
%     returns the distance after scaling each dimension by a factor of
%     0.5, which gives 1.
%   weightedDist([0;0], [1;2], [1 0.5].^2)
%     returns 2 (scaling dimension 1 by 1, dimension 2 by 0.5)
%   
% See also: distC,mahalanobis_dist
% 

% Author(s), Copyright: Anton Schwaighofer, Mar 2005
% $Id: weightedDist.m,v 1.5 2006/06/19 19:59:14 neuro_toolbox Exp $

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

% Compute the weighted squared distances between X1 and X2
X11 = (X1.*X1)'*beta;
if isempty(X2),
  X11mult = X11*ones([ndims N1]);
  D = X11mult+X11mult' - 2*X1'*beta*X1;
else
  X22 = beta*(X2.*X2);
  D = X11*ones([ndims N2]) - 2*X1'*beta*X2 + ones([N1 ndims])*X22;
end
% Rounding errors occasionally cause negative entries in D. This is
% problematic with the Matern kernel function, but not in other kernel
% functions
if any(any(D<0))
  D(D<0) = 0;
end
% There are some bizarre cases where the diagonal elements become
% non-zero, e.g. with some inweights extremely high. Correct that if
% possible (i.e., with empty X2 if we know for sure that the elements
% must be zero)
if isempty(X2),
  D(1:N1+1:N1*N1) = 0;
end
