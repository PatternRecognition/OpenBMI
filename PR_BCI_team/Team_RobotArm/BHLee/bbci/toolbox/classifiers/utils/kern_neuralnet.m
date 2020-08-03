function [K diagK] = kern_neuralnet(X1,X2,varargin)
% kern_neuralnet - Neural network kernel (Williams 1998 NC)
%
% Synopsis:
%   K = kern_neuralnet(X1,X2)
%   K = kern_neuralnet(X1,X2,'Property',Value,...)
%   [K diagK] kern_neuralnet(X1,X2,'Property',Value,...)
%   
% Arguments:
%   X1: [d N1] matrix. Input data 1 for kernel matrix computation
%   X2: [d N2] matrix. Input data 2 for kernel matrix computation. If
%       left empty, X2==X1 will be assumed.
%   
% Returns:
%   K: [N1 N2] matrix with the evaluation of the kernel function for all
%      pairs of points X1 and X2.
%  dK: [N1 1] diagonal of the square kernel matrix evaluated at X1
%   
% Properties:
%   'inweights': Scalar or [1 d] vector. (Squared) weights for each input
%      dimension. This parameter is given on log-scale. Default:
%      log(1/d), where d is the number of input dimensions.
%   'bias': Scalar, variance of the bias term, in log scale
%
% Description:
%   The neural network kernel is given by
%     k(x1,x2) = a * asin( (b+z12)/(sqrt(1+b+z11) * sqrt(1+b+z22)) )
%   where z_nm is the weighted inner product between x_n and x_m,
%     z_nm = sum_{i=1}^d x_n(i) * x_m(i) * inweight(i)
%
% NOTE: if both K and diagK are requested, it means that *only* the
%       diagonal of K at X1 is wanted, which is put in diagK
%   
% Examples:
%   X1 = [2 2; 3 4]'; X2 = [2 2; 2 3; 4 4]';
%   kern_neuralnet(X1)
%     evaluates the NEURALNET kernel with its default parameter settings on all
%     pairs of points (pairs of columns) in X1.
%   kern_neuralnet(X1, X2)
%     does the same for points in X1 and X2
%   kern_neuralnet(X1, [], 'inweights', 1)
%     uses exp(1) as the input weight across all dimensions,
%   kern_neuralnet(X1, [], 'inweights', [0 1])
%     uses exp(0) as the weight along the first dimension, exp(1) along
%     the second.
%   
% See also: kernderiv_neuralnet
% 
% Author(s): Joaquin Quiñonero Candela, March 2006
% $Id: kern_neuralnet.m,v 1.1 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(1, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'inweights', ones([1 ndims])*log(1/ndims));
opt = set_defaults(opt, 'bias', log(1));

scalarInweights = prod(size(opt.inweights))==1;

if nargout == 1,                        % want the kernel
  if isempty(X2)                          % if X2 not given, then take X2 = X1
    D = weightedInnerProd(X1,X1,exp(opt.inweights));
    Z = (exp(opt.bias)+D)./(sqrt(1+exp(opt.bias)+diag(D))*sqrt(1+exp(opt.bias)+diag(D)'));
    K = asin(Z);
  else                                    % if X2 given
    d1 = weightedDist(X1,zeros([ndims 1]),exp(opt.inweights));
    d2 = weightedDist(X2,zeros([ndims 1]),exp(opt.inweights));
    D = weightedInnerProd(X1,X2,exp(opt.inweights));
    Z = (exp(opt.bias)+D)./(sqrt(1+exp(opt.bias)+d1)*sqrt(1+exp(opt.bias)+d2'));
    K = asin(Z);
  end
elseif nargout == 2,                    % want the diagonal only, ignore X2
  K = [];               % don't waste efforts computig this, we only want
                        % the diagonal
  D = weightedSquareNorm(X1,exp(opt.inweights));
  Z = (exp(opt.bias)+D)./(1+exp(opt.bias)+D);
  diagK = asin(Z);
end

