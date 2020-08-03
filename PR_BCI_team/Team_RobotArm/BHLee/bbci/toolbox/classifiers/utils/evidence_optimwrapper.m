function [e, gradE] = evidence_optimwrapper(paramVect, f, X, y, opt, varargin)
% evidence_optimwrapper - Helper function for optimizing parameters in Gaussian Process models
%
% Synopsis:
%   e = evidence_optimwrapper(paramVect,f,X,y,opt)
%   [e,gradE] = evidence_optimwrapper(paramVect,f,X,y,opt)
%   [e,gradE] = evidence_optimwrapper(paramVect,f,X,y,opt,'Property',Value,...)
%   
% Arguments:
%  paramVect: Vector, containing all model and kernel parameters
%  f: String, name of the objective function
%  X: [dim N] matrix, training data
%  y: [nClasses N] matrix, regression target values or class labels of
%      the training data
%  opt: Struct. Gaussian Process model options, as set up in
%      train_GaussProc or train_classifierGP
%   
% Returns:
%  e: Scalar, value of the objective function
%  gradE: [nParams 1] vector, gradients of the objective function wrt all
%      model and kernel parameters
%   
% Description:
%   Wrapper function that only 
%   - copies all parameters from the given paramVect to the options
%     structure
%   - calls the specified objective function with the given training data
%   
%   
% See also: 
%   gaussProc_packParams,gaussProc_unpackParams
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2006
% $Id: evidence_optimwrapper.m,v 1.2 2006/06/19 19:59:14 neuro_toolbox Exp $


% From the caller we get the current point of optimization (paramVect)
% and the global options structure. Copy relevant parts from point of
% optimization to options
opt = gaussProc_unpackParams(paramVect, opt);

if nargout==2,
  [e, gradE] = feval(f, X, y, opt, varargin{:});
else
  e = feval(f, X, y, opt, varargin{:});
end
