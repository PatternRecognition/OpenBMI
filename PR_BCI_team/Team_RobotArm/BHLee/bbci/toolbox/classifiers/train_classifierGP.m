function net = train_classifierGP(Xtrain, Y, varargin)
% train_classifierGP - Gaussian Process classification model
%
% Synopsis:
%   net = train_classifierGP(Xtrain,Ytrain)
%   net = train_classifierGP(Xtrain,Ytrain,Property,Value,...)
%   
% Arguments:
%   Xtrain: [d N] matrix of training data (N points in d dimensions)
%   Ytrain: [2 N] matrix, indicators for class membership. Ytrain(1,i)==1
%      means that example X(:,i) belongs to class 1, Ytrain(2,i)==1 means
%      class 2.
%      The classifier output (apply_classifierGP) returns the probability
%      for membership in class 2. (chosen for consistency with other
%      classifiers where small value means class 1, large means class 2)
%   
% Returns:
%   net: Trained GP model
%   
% Properties:
%   'kernel': String, cell array {'func', param, ...} or nested cell array
%      {{'func1', param, ...},{'func2', param}}
%      Kernel function to use, name is given by 'kern_' plus the string
%      passed here. Eventual parameters in the cell array are passed to
%      the kernel function. With nested cell array, use a linear
%      combination of the specified kernel functions. Default: 'rbf'
%  'kernelweight': [1 k] vector. Initial weights (on log scale) for each
%      of the kernel functions, when using a linear combination. Weights
%      will be optimized during training. Default value: uniformly
%      log(1/k) for each of the k kernels.
%  'kernelindex': [1 k] cell array. Kernels can operate on different
%      features of the data. .kernelindex{i} is the feature index for the
%      i.th kernel (subset X(kernelindex{i},:) of the data). If empty,
%      the corresponding kernel uses all features. Default value: {}
%  'optimizer': String or cell array {'func', param, ...}. Optimizer
%      routine to use when adapting kernel parameters. Name of the optimizer
%      is the string passed here, with optional parameters in the cell
%      array. If set to [], do not optimize hyper parameters at all.
%      Default value: 'opt_scaledConjGrad'.
%  'clamped': Cell array. List of model and kernel parameters that are
%      kept at fixed values during evidence maximization. Each entry in the
%      cell is either 'paramname' or {'paramname', dim} (see below). Default
%      value: {}
%  'scaledata': Logical. If true, try to scale the features (columns) of
%      the data to a reasonable range before training. Default: 1.
%  'featurefilter': String or cell array {'func', param, ...}. Function
%      to call when selecting feature subsets based on p-values of
%      correlation between target value and individual features. Default: {}
%  'featurethresh': Scalar, threshold to use for p-value based
%      filtering. Default: 0.05
%  'EPiterations: Scalar. Number of sweeps when computing EP
%      approximation. Default value: 20.
%  'EPtolerance: Scalar. Tolerance for EP approximation. Terminate if the
%      difference of natural parameter 2 is below the tolerance
%      value. Default value: 1e-4
%  'storeKtrain': Logical. If true, store inverse kernel matrix of
%      training data for later use in apply_GaussProc. This is only
%      useful if predictive variances are required in apply_GaussProc. 
%      Default value: 0
%  'verbosity': Scalar. If 0, display nothing. If 1, display some
%      information about training procedure. 2: Also display kernel
%      statistics and training progress. Default value: 2
%   
% Description:
%   This routine provides a Gaussian Process classification model with
%   probit likelihood. Training consists of fitting the model's kernel
%   parameters to the data, by maximizing evidence (i.e., marginal
%   likelihood). Here, the Expectation Propagation (EP) approximation is
%   used.
%
%   With some data, the EP approximation sometimes runs into trouble, by
%   choosing incredibly large kernel amplitudes. To avoid this, the GP
%   classifier sets a prior on the kernel amplitudes by default,
%   equivalently to specifying the option opt.prior = {'kernelweight', {0 4}}
%
% See also: apply_classifierGP,train_GaussProc,gaussProc_paramPrior
% 

% Author(s): Anton Schwaighofer, Aug 2005
% $Id: train_classifierGP.m,v 1.3 2007/09/24 16:05:34 neuro_toolbox Exp $

error(nargchk(2, Inf, nargin));

opt = propertylist2struct(varargin{:});
% Leave in a dummy noise param so that gaussProc_covariance works
[opt, isdefault] = set_defaults(opt, 'kernel', 'rbf', ...
                                     'kernelweight', [], ...
                                     'kernelindex', {}, ...
                                     'featurefilter', {}, ...
                                     'featurethresh', 0.05, ...
                                     'clamped', {}, ...
                                     'noise', log(1e-5), ...
                                     'minNoise', log(eps^(1/2)), ...
                                     'EPiterations', 20, ...
                                     'EPtolerance', 1e-4, ...
                                     'optimizer', {'opt_BFGS', 'verbosity', 1}, ...
                                     'scaledata', 1, ...
                                     'storeKtrain', 0, ...
                                     'verbosity', 2, ...
                                     'prior', {'kernelweight', {0 4}});

% With high verbosity, also let optimizer output its progress
if isdefault.optimizer,
  if opt.verbosity>=2,
    opt.optimizer = {opt.optimizer{:}, 'verbosity', 2};
  end
end
[dim, N] = size(Xtrain);
[nClasses, NY] = size(Y);
if NY~=N,
  error('Xtrain and Ytrain must have matching number of columns');
end

switch nClasses
  case 1
    % Row vector given: Allow 0/1 targets or +1/-1 target
    if isempty(setxor(unique(Y), [0 1])),
      if opt.verbosity>1,
        fprintf('Targets are coded as 0/1.\n');
      end
      Ytrain = Y*2-1;
    elseif isempty(setxor(unique(Y), [-1 1])),
      if opt.verbosity>1,
        fprintf('Targets are coded as -1/+1.\n');
      end
      Ytrain = Y;
    elseif length(unique(Y))==1,
      error('The training data seems to consist of only one class?');
    else
      error('Matrix Ytrain must contain either [-1 +1] or [0 1] values');
    end
  case 2
    % Ytrain are targets given as unit vectors: Assume this is a
    % classification task with nClasses
    if ~isempty(setxor(double(unique(Y)), [0 1])),
      error('Matrix Ytrain must only contain 0/1 values (unit vectors)');
    end
    if opt.verbosity>1,
      fprintf('Targets are coded as unit vectors.\n');
    end
    hasClass = any(Y,1);
    if any(~hasClass),
      if opt.verbosity>0,
        fprintf('Discarding %i samples that are not assigned to any class.\n', ...
                nnz(~hasClass));
      end
      Y = Y(:,hasClass);
      Xtrain = Xtrain(:,hasClass);
    end
    % Convert to +1/-1
    Ytrain = [-1 +1]*Y;
  otherwise
    error('Input Ytrain must have no more than 2 rows');
end

if any(any(isnan(Xtrain))) | any(isnan(Ytrain)),
  error('Training data must not contain NaN values');
end
% If requested, remove features that are uncorrelated with target value
[Xtrain,opt,selected] = gaussProc_featureFilter(Xtrain,Ytrain,opt);
dim = size(Xtrain,1);

% Make a range check for the input. If features are badly scaled, we can
% expect trouble with optimization
[Xtrain,featureMean,featureStd] = gaussProc_scaleData(Xtrain, opt);

% If the optimizer has been manually set to [], we do not do any
% optimization at all.
if ~isempty(opt.optimizer),
  [optFunc, optParam] = getFuncParam(opt.optimizer);

  % Consistency checks for kernels. Also extract parameters from each
  % kernel function for later optimization
  opt = gaussProc_prepareKernels(Xtrain, opt, 1);
  
  nKernels = length(opt.kernel);
  for i = 1:nKernels,
    kernelOpt = opt.kernel{i}{2};
    % Exclude all parameters from optimization that should be kept constant
    % (opt.clamped)
    kernelOpt.allParams = gaussProc_removeClamped(kernelOpt.allParams, opt.clamped);
    opt.kernel{i} = {opt.kernel{i}{1}, kernelOpt};
  end
  
  % Similarly, prepare actual model parameters, so that struct2vect can be
  % used here as well. Kernel parameters and noise model parameters (c and d
  % of the Gamma distribution) will be optimized by a numerical optimizer.
  modelParams = {};
  for i = 1:nKernels,
    modelParams{end+1} = {'kernelweight', i};
  end
  opt.allParams = gaussProc_removeClamped(modelParams, opt.clamped);
  
  paramVect = gaussProc_packParams(opt);
  [paramVect, fOpt] = feval(optFunc, {'evidence_optimwrapper', ...
                      'evidence_classification', Xtrain, Ytrain, opt}, ...
                            paramVect, optParam{:});
  opt = gaussProc_unpackParams(paramVect, opt);
  % After optimization, we don't need this field anymore (only used
  % internally for computing derivatives)
  for i = 1:nKernels,
    opt.kernel{i}{2} = rmfield(opt.kernel{i}{2}, 'allParams');
  end
  opt = rmfield(opt, 'allParams');
else
  % Consistency checks for kernels, but don't prepare params for optimization
  opt = gaussProc_prepareKernels(Xtrain, opt, 0);
  fOpt = NaN;
  if opt.verbosity>0,
    fprintf('Option ''optimizer'': No optimizer given. Leaving all parameters at their chosen values.\n');
  end
end

% Need to recompute the EP site parameters here for prediction:
K = gaussProc_evalKernel(Xtrain, [], opt);
[nat1Site, nat2Site, nat1Cavity, nat2Cavity,invL,evidence] = ...
    gaussProc_classificationEP(K,Ytrain,opt);
% Precomputed kernel weights for prediction:
v = sqrt(nat1Site(:));
invKS = (v*v').*(invL'*invL);

net = struct('type', 'classifierGP');
net.opt = opt;
net.evidencePerCase = evidence/N;
net.filteredFeatures = selected;
net.featureMean = featureMean;
net.featureStd = featureStd;
net.nat1Site = nat1Site;
net.nat2Site = nat2Site;
net.nat1Cavity = nat1Cavity;
net.nat2Cavity = nat2Cavity;
if opt.storeKtrain,
  net.invKS = invKS;
end
net.alpha = invKS * (nat2Site./nat1Site);
% Also store the training data in the GP (needed later for computing the
% kernels for test data)
net.Xtrain = Xtrain;
% Ytrain are actually not needed, still store them
net.Ytrain = Ytrain;

% This is a hack for the wdpos-kernel. After training, delete the helper structure for each subkernel.
for i = 1:length(opt.kernel)
    if (isfield(net.opt.kernel{i}{2}, 'matchMatrices'))
	net.opt.kernel{i}{2} = rmfield(net.opt.kernel{i}{2}, 'matchMatrices');
    end
end
