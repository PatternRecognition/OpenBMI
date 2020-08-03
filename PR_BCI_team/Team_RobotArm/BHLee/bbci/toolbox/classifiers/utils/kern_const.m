function K = kern_const(X1,X2,varargin)
% kern_rbf - Radial basis function RBF kernel (Gaussian kernel)
%
% Synopsis:
%   K = kern_rbf(X1,X2)
%   K = kern_rbf(X1,X2,'Property',Value,...)
%   
% Arguments:
%   X1: [d N1] matrix. Input data 1 for kernel matrix computation
%   X2: [d N2] matrix. Input data 2 for kernel matrix computation. If
%       left empty, X2==X1 will be assumed.
%   
% Returns:
%   K: [N1 N2] matrix with the evaluation of the kernel function for all
%      pairs of points X1 and X2.
%   
% Properties:
%   'inweights': Scalar or [1 d] vector. (Squared) weights for each input
%      dimension. This parameter is given on log-scale. Default:
%      log(1/d), where d is the number of input dimensions.
%
% Description:
%   The radial basis function kernel (Gaussian kernel, squared
%   exponential kernel) is given by
%     k(x1,x2) = exp(-0.5*z)
%   where z is the weighted distance between x1 and x2,
%     z = sum_{i=1}^d (x1(i)-x2(i))^2 * inweight(i)
%   
%   
% Examples:
%   X1 = [2 2; 3 4]'; X2 = [2 2; 2 3; 4 4]';
%   kern_rbf(X1)
%     evaluates the RBF kernel with its default parameter settings on all
%     pairs of points (pairs of columns) in X1.
%   kern_rbf(X1, X2)
%     does the same for points in X1 and X2
%   kern_rbf(X1, [], 'inweights', 1)
%     uses exp(1) as the input weight across all dimensions,
%   kern_rbf(X1, [], 'inweights', [0 1])
%     uses exp(0) as the weight along the first dimension, exp(1) along
%     the second.
%   
% See also: kernderiv_rbf
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: kern_const.m,v 1.1 2005/09/02 14:53:15 neuro_toolbox Exp $

error(nargchk(1, inf, nargin));
[ndims N1] = size(X1);
if nargin<2 | isempty(X2),
  N2 = N1;
else
  N2 = size(X2, 2);
end
K = ones([N1 N2]);
