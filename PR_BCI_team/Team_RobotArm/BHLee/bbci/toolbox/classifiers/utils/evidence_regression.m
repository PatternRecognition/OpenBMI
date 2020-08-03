function [e, gradE] = evidence_regression(X,Y,opt)
% evidence_regression - Marginal likelihood (evidence) for Gaussian process regression model
%
% Synopsis:
%   e = evidence_regression(X,Y,opt)
%   [e, gradE] = evidence_regression(X,Y,opt)
%   
% Arguments:
%   X: [d N] matrix. Input data, N points in d dimensions
%   Y: [1 N] matrix. Regression target values
%  opt: Struct array. This contains all options for a Gaussian process
%      regression model, as used in train_GaussProc. The fields opt.noise,
%      opt.minNoise (noise variance and minimum noise variance, both given on
%      log scale), and opt.kernel are mandatory. Also, opt.kernel{2} needs
%      to have a field 'allParams' containing the names of kernel
%      parameters (see kernel derivative routines, eg. kernderiv_rbf.m)
%   
% Returns:
%   e: Negative log-likelihood of the data X, Y, under the chosen
%       model. Constant (normalization) terms are omitted.
%   gradE: [n 1] matrix. Each entry contains the derivative of the
%       evidence wrt. one kernel or Gaussian process parameter
%   
% Description:
%   When optimizing kernel parameters in Gaussian process models, the
%   marginal likelihood is most often used as the objective
%   function. This routine computes the likelihood of a particular data
%   set in a simple Gaussian process regression model.
%   The routine relies on a correct setup of the options structure, use
%   outside of train_GaussProc should be handled with care.
%
%   The derivatives of the evidence wrt. kernel parameters are computed
%   by calling kernel derivative routines. The caller of this routine
%   needs to ensure that the kernel parameters, given in opt.kernel{2},
%   are given as a struct array. Also, this struct array needs to have a
%   field 'allParams', containing all kernel parameters for which a
%   derivative can be computed.
%
%   For reasons of lazyness, the contributions of parameter priors (see
%   function gaussProc_paramPrior) are only computed when specifying 2
%   output arguments, that is, if gradients are requested.
%   
% Examples:
%   Setting up the required fields:
%     [kernelFunc, param] = getFuncParam(opt.kernel);
%     [allParams,kernelOpt] = ...
%       feval(['kernderiv_' kernelFunc], Xtrain, [], 'all', param{:});
%     kernelOpt.allParams = allParams;
%     opt.kernel = {kernelFunc, kernelOpt};
%   Computing the actual gradient:
%     [e,gradE] = evidence_regression(X, Y, opt);
%   
%   
% See also: train_GaussProc
% 

% Author(s): Anton Schwaighofer, Sep 2005
% $Id: evidence_regression.m,v 1.3 2007/08/10 18:17:43 neuro_toolbox Exp $

error(nargchk(3, 3, nargin));

[K,indivK] = gaussProc_covariance(X, opt);

infK = isinf(K);
nanK = isnan(K);
% For those cases where optimization fails completely, and weights or cov
% function parameters are totally out of range. Evaluate all(all(..))
% before 'or'ing the matrices
if all(all(infK)) | all(all(nanK)),
  fckedup = 1;
  e = Inf;
else
  % The not so awful cases: Only some NaN entries. Solve with a quick'n'dirty hack...
  fckedup = 0;
  K(nanK) = realmax;
  K(infK & (K<0)) = -realmax;
  K(infK & (K>0)) = realmax;
  [invK, logdetK, q] = inv_logdet_pd(K);
  if q>0,
    % Cholesky failed: Stick to a more time consuming solution. This may
    % happen eg. for Rational Quadratic kernels with very large degrees,
    % or for the Matern kernel
    eigK = eig(K, 'nobalance');
    % Guard against possible tiny negative eigenvalues (eg. in the Matern
    % kernel with large values of nu)
    if any(eigK<=0),
      warning('Skipping some negative eigenvalues. Results may be inaccurate');
    end
    logdetK = sum(log(eigK(eigK>0)));
    invK = inv(K);
  else
    fckedup = 0;
  end
  e = 0.5*(logdetK+Y*invK*Y');
end
% So far, this is unnormalized. For easier interpretation, switch to
% normalized (negative) evidence per case
[dim N] = size(X);
e = (N/2*log(2*pi) + e)/N;


if nargout<2,
  return;
end

nKernels = length(opt.kernel);
N = 0;
% First compute the total length of the parameter gradient vector
for i = 1:nKernels,
  N = N+length(opt.kernel{i}{2}.allParams);
end
% Pre-allocate gradient vector
gradE = zeros([N+length(opt.allParams) 1]);
% Start writing into g at 1
start = 1;
% Summing up all the contributions from the parameter prior to the
% evidence
prior_e = 0;

if fckedup~=0,
  gradE = gradE*Inf;
  return;
end

traceInvK = trace(invK);
invKt = invK*Y';

KKt = invK-invKt*invKt';

for i = 1:nKernels,
  [kernelFunc, kernelParam] = getFuncParam(opt.kernel{i});
  % In the first step, where we called the kernderiv routine, we should
  % have obtained kernel parameters packed into a struct, there is no longer
  % a property/value list
  kernelParam = kernelParam{1};
  % Derivative routines often need the kernel matrix, pass as option to
  % avoid re-computation
  kernelParam.K = indivK{i};
  % For the rational quadratic and the neural network kernels, we need to pass
  % pre-computed things between the subsequent calls. This flag here can be
  % used by the routine to set everything up at the first iteration
  kernelParam.resetTransient = 1;
  % Evaluate derivatives with respect to each kernel hyperparameter in turn.
  for j = 1:length(kernelParam.allParams),
    % The field allParams stores a cell array of all derivatives the kernel
    % function can compute
    p = kernelParam.allParams{j};
    % Get the matrix of derivatives with respect to that particular
    % parameter. Kernel derivative routines can modify the parameter set,
    % in order to pass options between subsequent calls (eg in the Matern
    % kernel). This is not a very clean solution, but the best I could
    % come up with.
    if isempty(opt.kernelindex) | isempty(opt.kernelindex{i}),
      [kernDeriv, kernelParam, scaling] = ...
          feval(['kernderiv_' kernelFunc], X, [], p, kernelParam);
    else
      [kernDeriv, kernelParam, scaling] = ...
          feval(['kernderiv_' kernelFunc], X(opt.kernelindex{i},:), [], p, kernelParam);
    end
    scaling = scaling*exp(opt.kernelweight(i));
    % Derivatives of the evidence with respect to the kernel parameters:
    gradE(start) = 0.5*scaling*(sum(sum(KKt.*kernDeriv)));
    % Possibly we also have a contribution from the parameter prior:
    [prior_e0, prior_gradE] = gaussProc_paramPrior(p, kernelParam);
    prior_e = prior_e + prior_e0;
    gradE(start) = gradE(start) + prior_gradE;
    start = start+1;
  end
end
for i = 1:length(opt.allParams),
  p = opt.allParams{i};
  switch p{1}
    case 'noise'
      % Derivative with respect to noise variance
      if isempty(opt.noisegroups),
        % One shared noise variance:
        gradE(start) = 0.5*exp(opt.noise)*(traceInvK-invKt'*invKt);
      else
        dim = p{2};
        ind = opt.noisegroups{dim};
        % We have noise elements along the diagonal of S in
        % K+S. Differentiating leaves ones at those positions that
        % correspond to the noise parameter we are currently
        % differentiating. Derivative term contains trace(invK * gradS),
        % with gradS being diagonal with many zeros. Thus, we can as well
        % take subsets of invK. The same goes for the invKt terms
        traceInvK2 = trace(invK(ind,ind));
        gradE(start) = 0.5*exp(opt.noise(dim))*(traceInvK2-invKt(ind)'*invKt(ind));
      end
    case 'kernelweight'
      % Derivative of exp(kernelweight)*kernel gives the very same term
      dim = p{2};
      weightDeriv = exp(opt.kernelweight(dim))*indivK{dim};
      gradE(start) = 0.5*(sum(sum(KKt.*weightDeriv)));
    otherwise
      error(sprintf('Unknown parameter %s', opt.allParams{i}));
  end
  % Possibly we also have a contribution from the parameter prior:
  [prior_e0, prior_gradE] = gaussProc_paramPrior(p, opt);
  prior_e = prior_e + prior_e0;
  gradE(start) = gradE(start) + prior_gradE;
  start = start+1;
end
% For easier interpretation, switch to normalized (negative) evidence per
% case
gradE = gradE/size(X,2);
% Prior contribution to the marginal likelihood
e = e + prior_e/size(X,2);
