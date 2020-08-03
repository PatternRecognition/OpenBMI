function [e, gradE] = evidence_classification(X, y, opt)
% evidence_classification - Marginal likelihood (evidence) for GP classification model
%
% Synopsis:
%   e = evidence_classification(X,Y,opt)
%   [e, gradE] = evidence_classification(X,Y,opt)
%   
% Arguments:
%   X: [d N] matrix. Input data, N points in d dimensions
%   Y: [1 N] matrix. Regression target values
%  opt: Struct array. This contains all options for a Gaussian process
%      regression model, as used in train_classifierGP. The field
%      opt.kernel is mandatory. Also, opt.kernel{i}{2} needs
%      to have a field 'allParams' containing the names of kernel
%      parameters (see kernel derivative routines, eg. kernderiv_rbf.m)
%   
% Returns:
%   e: Negative marginal log-likelihood of the data X, Y, under the chosen
%       model. Constant (normalization) terms are omitted.
%   gradE: [n 1] matrix. Each entry contains the derivative of the
%       evidence wrt. one kernel or Gaussian process parameter
%   
% Description:
%   Approximate marginal likelihood of the GP classification model with
%   probit likelihood, approximation via expectation propagation (EP)
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
%   
% See also: train_classifierGP,gaussProc_classificationEP
%

% Anton Schwaighofer <anton@first.fraunhofer.de>
% Inspired by code from Malte Kuss <malte.kuss@tuebingen.mpg.de>

[K,indivK] = gaussProc_evalKernel(X, [], opt);

[dim, m] = size(X);

% Run Expectation Propagation algorithm and compute the negative log-evidence
[nat1Site, nat2Site, nat1Cavity, nat2Cavity,invL,evidence] = ...
    gaussProc_classificationEP(K,y,opt);
% For easier interpretation, switch to (negative) evidence per case
e = evidence/m;

%
% Compute the gradients
%

if nargout<2,
  return;
end

% Summing up all the contributions from the parameter prior to the
% evidence
prior_e = 0;
% invKS = diag(sqrt(nat1Site)) * invL' * invL * diag(sqrt(nat1Site));
v = sqrt(nat1Site(:));
invKS = (v*v').*(invL' * invL);
b = nat2Site - invKS*(K*nat2Site);
KW = (invKS - b*b');

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

for i = 1:nKernels,
  [kernelFunc, kernelParam] = getFuncParam(opt.kernel{i});
  % In the first step, where we called the kernderiv routine, we should
  % have obtained kernel parameters packed into a struct, there is no longer
  % a property/value list
  kernelParam = kernelParam{1};
  % Derivative routines often need the kernel matrix, pass as option to
  % avoid re-computation
  kernelParam.K = indivK{i};
  % For the rational quadratic kernel, we need to pass pre-computed
  % things between the subsequent calls. This flag here can be used by
  % the routine to set everything up at the first iteration
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
    % 0.5*trace(kernDeriv*KW), rewrite as sum(sum(...))
    gradE(start) = 0.5*scaling*sum(sum(KW.*kernDeriv));
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
    case 'kernelweight'
      % Derivative of exp(kernelweight)*kernel gives the very same term
      dim = p{2};
      gradE(start) = 0.5*exp(opt.kernelweight(dim))*sum(sum(KW.*indivK{dim}));
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
gradE = gradE/m;
% Prior contribution to the marginal likelihood
e = e + prior_e/m;
