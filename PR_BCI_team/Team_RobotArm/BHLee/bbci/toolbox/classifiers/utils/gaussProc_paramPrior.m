function [e,grade] = gaussProc_paramPrior(whichParam,varargin)
% gaussProc_paramPrior - Generic function for GP model and kernel parameters priors
%
% Synopsis:
%   [e,grade] = gaussProc_paramPrior(whichParam, 'Property',Value,...)
%   
% Arguments:
%  whichParam: String or cell. Kernel parameter for which likelihood and gradient
%     should be computed. If given as cell, the cell contents is assumed
%     to be {'name', dimension}
%   
% Returns:
%  e: Scalar. Negative log likelihood of the kernel parameters under the prior
%  grade: Scalar. Gradient of the parameter negative log likelihood wrt the given
%      parameter
%   
% Description:
%   Prior likelihoods for parameters of a Gaussian Process model, this can
%   be either model parameters or kernel parameters.  Currently, only normal
%   distributions are implemented as possible priors, resp. log-normal
%   priors for parameters that are specified on log-scale (such as kernel
%   length scales or noise variances)
%   
%   Both parameters and prior specification are part of the function's
%   options, as specified by the property/value list. First, the options
%   are checked for field named 'prior'. Format of this field/property is
%   {'param1name', {param1mean,param1variance}, 'param2name', {param2mean,param2variance}}
%   
%   Field 'prior' is then scanned for the entry corresponding to the
%   parameter passed in argument 'whichParam'. Output e is the
%   log-likelihood of the parameter value under the specified prior
%   distribution. Currently, constant (scaling) terms are omitted in the computation
%   
% Examples:
%   opt.param1 = -5;
%   opt.prior = {'param1', {-4, 2}};
%   gaussProc_paramPrior('param1', opt)
%     returns 1.5155, which is the value of the normal PDF -log(normal_pdf(-5,-4,sqrt(2)))
%
%   Similarly, this can be used for parameter vectors:
%   opt.param2 = [-1 -2 -3];
%   opt.prior = {'param1', {-4, 2}, 'param2', {[0 0 2], 4}};
%   gaussProc_paramPrior({'param2',3}, opt)
%     returns 4.7371, the prior likelihood for parameter param2(3).
%   Mind that the prior for 'param2' is here specified with a shared
%   variance.
%     
%   
% See also: kern_rbf,evidence_regression
% 

% Author(s), Copyright: Anton Schwaighofer, May 2006
% $Id: gaussProc_paramPrior.m,v 1.1 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(1, inf, nargin));
opt = propertylist2struct(varargin{:});

e = 0;
grade = 0;
% Prior is stored in field 'prior', this can again be a property/value
% list
if ~isfield(opt, 'prior') | isempty(opt.prior),
  return;
end
prior = propertylist2struct(opt.prior{:});

% Dissect the argument into the string name of the parameter and the
% dimension (if any)
if iscell(whichParam),
  param = whichParam{1};
  if length(whichParam)>1,
    dim = whichParam{2};
  else
    dim = 1;
  end
else
  param = whichParam;
  dim = 1;
end

if ~isfield(opt, param),
  % Kernel function does not seem to have a parameter of that name
  error('Invalid parameter given in argument ''whichParam''');
elseif ~isfield(prior, param),
  % We have no information about a prior for this parameter: Do nothing
  return
end

% Prior must be specified as {mean, variance}, both can be arrays of
% different length
prior_spec = getfield(prior, param);
prior_mean = prior_spec{1};
if length(prior_mean)>1,
  prior_mean = prior_mean(dim);
end
prior_var = prior_spec{2};
if length(prior_var)>1,
  prior_var = prior_var(dim);
end

% Actual parameter value:
param_value = getfield(opt, param);
param_value = param_value(dim);
% Negative Gaussian log likelihood
e = ((param_value-prior_mean).^2)./(2*prior_var);
% The normalizing terms
e = e+(length(e)*log(2*pi)+log(prior_var))/2;
% ... and its derivative
grade = (param_value-prior_mean)./prior_var;
