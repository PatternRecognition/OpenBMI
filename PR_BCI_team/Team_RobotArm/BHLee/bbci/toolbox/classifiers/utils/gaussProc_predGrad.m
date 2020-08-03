function [grad,gradNames] = gaussProc_predGrad(net,Xtest)
% gaussProc_predGrad - Gradient of Gaussian process prediction wrt model parameters
%
% Synopsis:
%   grad = gaussProc_predGrad(GP,Xtest)
%   [grad,gradNames] = gaussProc_predGrad(GP,Xtest)
%   
% Arguments:
%  GP: Struct array. Trained Gaussian process model, as returned by
%      train_GaussProc. In particular, the fields .Xtrain, .alpha, and .opt
%      are required.
%  Xtest: [d N] matrix of test data (points on which to compute the GP
%      gradient)
%   
% Returns:
%  grad: [m N] matrix. grad(:,i) contains the GP model prediction and its
%      gradients at point Xtest(:,i)
%  gradNames: [1 m] cell array, label for the derivative returned in each
%      row of grad
%   
% Description:
%   This routine computes the gradient of a Gaussian process prediction
%   wrt to all model parameters. This is similar in spirit to the TOP
%   kernel (Tsuda et al, 2002), just that GP models are used.
%   The resulting gradient vector can be used in discriminative models.
%
%   Mind that heavy overfitting is possible when doing classification
%   with these gradient vectors. Subsequent classifiers should be linear
%   and restricted in discriminative power, eg. a SVM with *low* values
%   of C (eg. in the range of 10^6 through 1)
%   
%   
% References:
%   Koji Tsuda, Motoaki Kawanabe, Gunnar R{\"a}tsch, S{\"o}ren
%   Sonnenburg, and Klaus-Robert M{\"u}ller: A New Discriminative Kernel
%   From Probabilistic Models. Neural Computation, vol 14(10), pp
%   pp 2397--2414 (2002)
%
% See also: train_GaussProc
% 

% Author(s), Copyright: Anton Schwaighofer, May 2005
% $Id: gaussProc_predGrad.m,v 1.1 2005/05/25 08:41:04 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

if ~isfield(net, 'Xtrain') | ~isfield(net, 'Ytrain'),
  error('GP model must have fields ''Xtrain'' and ''Ytrain''');
end
% For gradients, we need the inverse training kernel matrix, plus the
% contributions of the individual kernel functions
[Knoise, K] = gaussProc_covariance(net.Xtrain, net.opt);
net.invKtrain = inv(Knoise);

net.alpha = net.invKtrain*net.Ytrain';

% Prepare for gradient computation: Extract list of model and kernel
% parameters
opt = net.opt;
nKernels = length(opt.kernel);
kernelParams = cell([1 nKernels]);
for i = 1:nKernels,
  % First, we need to extract the list of all kernel parameters for gradient
  % computation
  [kernelFunc, kernelParam] = getFuncParam(opt.kernel{i});
  % The kernel derivative routines need to return a list of all kernel
  % parameters. Also, pass kernel parameters to this function, so that we
  % can use the expanded options structure later (second return arg)
  if isempty(opt.kernelindex) | isempty(opt.kernelindex{i}),
    % No index given at all or nothing given for this particular
    % kernel: assume they operate on all data
    [kernelParams{i}, kernelOpt] = ...
        feval(['kernderiv_' kernelFunc], net.Xtrain, [], 'returnParams', kernelParam{:});
  else
    % Index given:
    [kernelParams{i}, kernelOpt] = ...
        feval(['kernderiv_' kernelFunc], net.Xtrain(opt.kernelindex{i},:), [], ...
              'returnParams', kernelParam{:});
  end
  % Replace the old entry for the kernel by the fully expanded options
  opt.kernel{i} = {kernelFunc, kernelOpt};
end
% Similarly, prepare actual model parameters. Currently, model parameters
% are 'noise' and the kernel weights. Use double cell for noise here, so that
% all params have unified format
modelParams = {{'noise'}};
for i = 1:nKernels,
  modelParams{end+1} = {'kernelweight', i};
end

% First compute the total length of the gradient matrix
m = 1; % contribution of GP output
for i = 1:nKernels,
  m = m+length(kernelParams{i});
end
m = m + length(modelParams);
% Pre-allocate gradient matrix
grad = zeros([m size(Xtest,2)]);
gradNames = cell([1 m]);

[kTestAll, kTest] = gaussProc_evalKernel(Xtest, net.Xtrain, net.opt); 

start = 1;
grad(start,:) = apply_GaussProc(net, Xtest);
gradNames{start} = 'GP model output';
start = start+1;
for i = 1:nKernels,
  [kernelFunc, kernelParam] = getFuncParam(opt.kernel{i});
  % Derivative routines often need the kernel matrix, pass as option to
  % avoid re-computation. Keep two copies of the kernelParams, in case
  % the gradient routines modify parameters separately when running on
  % the test and training data
  paramTrain = kernelParam{1};
  paramTrain.K = K{i};
  paramTest = kernelParam{1};
  paramTest.K = kTest{i};
  for j = 1:length(kernelParams{i}),
    theDeriv = kernelParams{i}{j};
    if isempty(opt.kernelindex) | isempty(opt.kernelindex{i}),
      [kTestDeriv, paramTest, scalingTest] = ...
          feval(['kernderiv_' kernelFunc], Xtest, net.Xtrain, theDeriv, paramTest);
      [kTrainDeriv, paramTrain, scalingTrain] = ...
          feval(['kernderiv_' kernelFunc], net.Xtrain, [], theDeriv, paramTrain);
    else
      [kTestDeriv, paramTest, scalingTest] = ...
          feval(['kernderiv_' kernelFunc], Xtest(opt.kernelindex{i},:), ...
                net.Xtrain(opt.kernelindex{i},:), theDeriv, paramTest);
      [kTrainDeriv, paramTrain, scalingTrain] = ...
          feval(['kernderiv_' kernelFunc], net.Xtrain(opt.kernelindex{i},:), [], ...
                theDeriv, paramTrain);
    end
    % General expression for derivative of GP prediction:
    grad(start,:) = (scalingTest*(kTestDeriv*net.alpha) - ...
                     scalingTrain*(kTestAll*net.invKtrain*(kTrainDeriv*net.alpha)))';
    gradNames{start} = theDeriv;
    start = start+1;
  end
end


for i = 1:length(modelParams),
  p = modelParams{i};
  switch p{1}
    case 'noise'
      % Derivative with respect to noise variance
      kTestDerivAlpha = 0;
      kTrainDeriv = exp(opt.noise)*speye(size(Knoise));
    case 'kernelweight'
      % Derivative of exp(kernelweight)*kernel gives the very same term
      dim = p{2};
      kTestDerivAlpha = exp(opt.kernelweight(dim))*(kTest{dim}*net.alpha);
      kTrainDeriv = exp(opt.kernelweight(dim))*K{dim};
    otherwise
      error(sprintf('Unknown parameter %s', p));
  end
  grad(start,:) = (kTestDerivAlpha - kTestAll*net.invKtrain*(kTrainDeriv*net.alpha))';
  gradNames{start} = p;
  start = start+1;
end
