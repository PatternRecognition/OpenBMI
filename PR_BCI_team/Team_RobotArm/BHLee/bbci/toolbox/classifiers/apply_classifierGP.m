function [pPred, means, variances] = apply_classifierGP(net, X)
% apply_classifierGP - Apply Gaussian process classification model
%
% Synopsis:
%   pPred = apply_GaussProc(net,X)
%   [pPred,means,variances] = apply_GaussProc(net,X)
%   
% Arguments:
%  net: Struct. Trained Gaussian process model, as output by
%      train_classifierGP. Required fields are
%      nat1Site, nat2Site, nat1Cavity, nat2Cavity, Xtrain and
%      (optionally) invKS.
%  X: [d N] matrix. Data to evaluate the Gaussian process model on (N
%      points in d dimensions)
%   
% Returns:
%  pPred: [1 N] matrix. Predicted class membership probability for each
%      test point. This is membership in class 2 (if training data was
%      coded as classes 1, 2), equivalently membership in class +1 (if
%      training data was coded as classes -1, +1)
%  means: [1 N] matrix. Predictive mean of the latent function for each
%      test point.
%  variances: [1 N] matrix. Predictive variance of the latent function
%      for each test point.
%   
% Description:
%   This function evaluates the predictive distributions for a Gaussian
%   process classification model on a set of test data. 
%   
%   
% See also: train_classifierGP,gaussProc_evalKernel
% 

% Author(s), Copyright: Anton Schwaighofer, Aug 2005
% $Id: apply_classifierGP.m,v 1.2 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

% Check whether feature subsets have been selected before training. If
% yes, choose the same subset of test features
if isfield(net, 'filteredFeatures'),
  sel = net.filteredFeatures;
  if any(~sel),
    X = X(sel,:);
  end
end

% Check whether the user had requested an automatic scaling of the data
if net.opt.scaledata,
  X = diag(1./net.featureStd)*(X-repmat(net.featureMean, [1 size(X,2)]));
end
if ~isfield(net, 'Xtrain') | ~isfield(net, 'alpha'),
  error('GP model must have fields ''Xtrain'' and ''alpha''');
end

if ~isfield(net, 'invKS') | isempty(net.invKS),
  K = gaussProc_evalKernel(net.Xtrain, [], net.opt);
  N = size(net.Xtrain,2);
  v = sqrt(net.nat1Site(:));
  invL = chol2invChol(chol(eye(N) + (v*v').*K)');
  net.invKS = (v*v').*(invL'*invL);
end

% Compute kernel matrix
K = gaussProc_evalKernel(X, net.Xtrain, net.opt);
means = (K*net.alpha)';

% Predictive variances:
diagKtest = zeros([size(X,2) 1]);
for i = 1:size(X,2),
  diagKtest(i) = gaussProc_evalKernel(X(:,i), [], net.opt);
end
variances = (diagKtest - sum(K .* (K * net.invKS),2))';

pPred = probitIntegral(means,variances);

function p = probitIntegral(m,var)
% Integral of a Gaussian and a standard normal cdf

p =  0.5*erfc(-m./(sqrt(2)*sqrt(1+var)));

