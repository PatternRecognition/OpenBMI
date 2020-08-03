function K = kern_matern(X1,X2,varargin)
% kern_matern - Matern kernel function
%
% Synopsis:
%   K = kern_matern(X1,X2)
%   K = kern_matern(X1,X2,'Property',Value,...)
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
%      dimension. This parameter is given on log-scale. Default: 0
%   'smoothness': Scalar. Degree nu of the Matern function. For nu->0, the
%      Matern function becomes the Ornstein-Uhlenbeck kernel, for
%      nu->inf, it becomes the RBF kernel. This parameter is given on
%      log-scale. Default: 1.
%
% Description:
%   The Matern class of functions is a covariance function where the
%   degree of smoothness can be varied continuously by its parameter
%   nu. nu=0 gives infinitely rough sample paths, the Matern function
%   corresponds to the Ornstein-Uhlenbeck process in this case. 
%   nu->inf gives sample paths that are infinitely times differentiable
%   (infinitely smooth). For nu->inf, the Matern function corresponds to
%   the radial basis function ("Gaussian") kernel.
%   The kernel function is given by
%     k(x1,x2) = matern(smoothness, sqrt(z), variance)
%   where z is the weighted distance between x1 and x2,
%     z = sum_{i=1}^d (x1(i)-x2(i))^2 * inweight(i)
%   
%   
% Examples:
%   X1 = [2 2; 3 4]'; X2 = [2 2; 2 3; 4 4]';
%   kern_matern(X1)
%     evaluates the Matern kernel with its default parameter settings on
%     all pairs of points (pairs of columns) in X1.
%   kern_matern(X1, [], 'inweights', [0 1], 'smoothness', 2)
%     uses exp(0) as the weight along the first dimension, exp(1) along
%     the second, and a Matern degree of exp(2)
%   
% See also: kernderiv_matern,matern
% 

error(nargchk(1, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'inweights', zeros([1 ndims]), ...
                        'smoothness', 1);

D = weightedDist(X1, X2, exp(opt.inweights));
% Don't forget.... D are the *squared distances* !!!
K = matern(exp(opt.smoothness), sqrt(D), 1);
