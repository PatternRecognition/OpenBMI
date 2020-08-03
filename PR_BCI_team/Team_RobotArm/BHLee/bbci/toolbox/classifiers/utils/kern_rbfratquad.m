function K = kern_rbfratquad(X1,X2,varargin)
% kern_rbfrbfratquad - RBF and rational quadratic kernel with shared input weights
%
% Synopsis:
%   K = kern_rbfratquad(X1,X2)
%   K = kern_rbfratquad(X1,X2,'Property',Value,...)
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
%   'degree': Scalar. Exponent in the expression (1+z)^(-degree). This
%      parameter is given on log-scale. Default: 0
%
% Description:
%   The rational quadratic kernel function is given by
%     k(x1,x2) = (1+z)^(-degree)
%   where z is the weighted distance between x1 and x2,
%     z = sum_{i=1}^d (x1(i)-x2(i))^2 * inweight(i)
%   
% See also: kernderiv_rbfratquad
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: kern_rbfratquad.m,v 1.2 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(1, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'inweights', ones([1 ndims])*log(1/ndims), ...
                        'degree', 0, ...
                        'rbfweight', 0);

D1 = weightedDist(X1, X2, exp(opt.inweights-log(2)));
D2 = D1*exp(-opt.degree);
% $$$ D2 = weightedDist(X1, X2, exp(opt.inweights-opt.degree-log(4)));
% I want to have a kernel with variance 1, so I need to have a form 
% weight*rbf+(1-weight)*ratquad. Squeeze weight for the RBF through a
% sigmoid to stay in the (0,1) range
K = 1./(1+exp(-opt.rbfweight)) * exp(-D1) + ...
    1./(1+exp(opt.rbfweight)) * (D2+1).^(-exp(opt.degree));
