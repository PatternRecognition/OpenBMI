function [Knoise,indivK,K] = gaussProc_covariance(X,opt)
% gaussProc_covariance - Return full covariance matrix (kernel plus noise) in GP regression model
%
% Synopsis:
%   Knoise = gaussProc_covariance(X,opt)
%   [Knoise,indivK,K] = gaussProc_covariance(X,opt)
%   
% Arguments:
%   X: [d N] matrix. Input data, N points in d dimensions
%   opt: Struct array. Options for Gaussian process model, as used in
%       train_GaussProc. Fields opt.noise, opt.minNoise and opt.kernel are mandatory
%   
% Returns:
%   Knoise: [N N] matrix. Full covariance matrix of the data, that is sum of
%       kernel matrix plus noise covariance
%   indivK: Cell array of [N N] matrices. Kernel function contribution to the
%       covariance for each kernel individually.
%   K: [N N] matrix. Full covariance matrix of the data, yet without
%       noise covariance
%   
% Description:
%   This routine simply computes the sum of the kernel function,
%   evaluated on the given points, plus a diagonal noise covariance
%   matrix.
%   
%   
% See also: train_GaussProc
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: gaussProc_covariance.m,v 1.5 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

[K, indivK] = gaussProc_evalKernel(X, [], opt); 
N = size(X,2);
% Make up the diagonal elements for the noise variance matrix
% Ungrouped data (the standard case):
if isempty(opt.noisegroups),
  % With ungrouped data, I allow two cases: Scalar noise (same variance
  % for all data points) and vector noise with one element per data point
  % (this is convenient for the routines for robust noise)
  if prod(size(opt.noise))==1,
    noiseVar = exp(max(opt.minNoise, opt.noise))*ones([N 1]);
  elseif prod(size(opt.noise))==N,
    noiseVar = exp(max(opt.minNoise, opt.noise(:)));
  else
    error('Invalid size for noise variance');
  end
else
  % Data with grouped noise variance:
  noiseVar = zeros([N 1]);
  for i = 1:length(opt.noisegroups),
    noiseVar(opt.noisegroups{i}) = exp(max(opt.minNoise, opt.noise(i)));
  end
end
Knoise = K + spdiags(noiseVar, 0, N, N);
