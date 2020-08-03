function C = train_RDAdist(X, labels, varargin)
% TRAIN_RDADIST - Train RDA classifier, with reject option at apply
%
%   Regularized Discriminant Analysis, with ability to reject examples if
%   their log-likelihood is too low.
%
% Usage:
%   C = TRAIN_RDADIST(X, LABELS)
%   C = TRAIN_RDADIST(X, LABELS, OPTS)
%
% Input:
%   X: Data matrix, one point/sample per column
%   LABELS: 0/1 matrix indicating class membership.
%   (see DATAFORMATS)
%   OPTS: Options structure. Recognized options:
%   'lambda': regularization parameter LAMBDA for RDA. Morph between
%      QDA (LAMBDA=0) and LDA (LAMBDA=1). Default value: 1.
%   'gamma': regularization parameter GAMMA for RDA. Morph between
%     covariance matrix estimated from data (GAMMA=0) and spherical
%     covariance matrix (GAMMA=1).
%   'rejectBelow': Reject if log-likelihood is below this value. Default
%     value: -Inf (that is, no rejections at all).
%     'rejectBelow' may be set to 'fadeOut' (for 2-class only). This
%     multiplies the classifier output with the class conditional density
%     of the winning class. Purpose: obtain a smooth classifier output
%     for BCI. Output will be zero in regions far from the class centres.
%   'classProb': Use the values passed here as the a-priori class
%     probabilities. If omitted, the class probabilities will be computed
%     from the training data. 
%   'func': a function to use
%   'bias': a bias
%
% Output:
%   C: trained classifier
%
% Details:
%   Examples are rejected if the log-likelihood is below the threshold
%   value for all classes.
%
% References:
%   J.H. Friedman, Regularized Discriminant Analysis, Journal
%   of the Americal Statistical Association, vol.84(405), 1989.
%   
%   
%   See also TRAIN_RDAREJCT, TRAIN_RLDA,APPLY_RDADIST,TRAIN_RDA,DEMO_REJECT
%

% Copyright Anton Schwaighofer (2004)
% $Id: train_RDAdist.m,v 1.3 2004/09/10 13:08:00 neuro_toolbox Exp $

error(nargchk(2, Inf, nargin));

opt = propertylist2struct(varargin{:});
% Set default parameters
opt = set_defaults(opt, 'lambda', 1, ...
                        'gamma', 0.01, ...
                        'rejectBelow', -Inf, ...
                        'classProb', [], ...
                        'func',inline('sqrt(x)','x'),...
                        'bias', 0);
if opt.gamma<0 | opt.gamma>1,
  error('Regularization parameter GAMMA must be between 0 and 1');
end
if opt.lambda<0 | opt.lambda>1,
  error('Regularization parameter LAMBDA must be between 0 and 1');
end

% Make things work even when labels are given in +1/-1 format
if size(labels, 1) == 1,
  labels = [labels<0; labels>0]; 
end

nClasses= size(labels,1);
dim = size(X, 1);

if isstr(opt.rejectBelow),
  if ~strcmp(lower(opt.rejectBelow), 'fadeout'),
    error('Invalid value for option ''rejectBelow''');
  end
  if nClasses~=2,
    error('The ''fadeOut'' option can only be used for 2-class problems');
  end
end

if ~isempty(opt.classProb),
  % Class probabilities are given as options: Make consistency check
  if length(opt.classProb)~=nClasses | abs(sum(opt.classProb)-1)>eps,
    error('Class probabilites must be of length NCLASSES and sum to 1');
  end
end

% Training is equal to RLDA, rejecting only matters in the apply
% phase.

% We can directly use the labels as binary indicators to index those
% examples that belong to a particular class.
labels = labels~=0;

% Classifier structure with identification field. Options will be
% required in the apply phase, so store them
C = struct('type', 'RLDAreject');
C.gamma = opt.gamma;
C.lambda = opt.lambda;
C.rejectBelow = opt.rejectBelow;
% Extract the number of examples per class - easy since labels are now 0/1
perClass = sum(labels, 2);
% Use class probabilities from options or from training data
if ~isempty(opt.classProb),
  C.classProb = opt.classProb;
  missingExamples = (C.classProb>0) & (perClass==0);
  if any(missingExamples),
    error(sprintf('No examples given for class %i', find(missingExamples)));
  end
else
  C.classProb = perClass./sum(perClass);
end

% For RLDA, we need to store both the per-class mean and the per-class
% covariance matrix, as well as the pooled covariance matrix
C.classMean = zeros(dim, nClasses);
classCov = zeros(dim, dim, nClasses);
poolCov = zeros(dim, dim);
for k = 1:nClasses,
  % Avoid empty classes
  if perClass(k)>0,
    % Examples that fall into class k
    XperClass = X(:,labels(k,:));
    C.classMean(:,k) = mean(XperClass,2);
    % Subtract the class mean
    Xnorm = XperClass-repmat(C.classMean(:,k), [1 perClass(k)]);
    % This gives the unnormalized covariance matrix
    ssd = Xnorm*Xnorm';
    classCov(:,:,k) = ssd./(perClass(k)-1);
    % Also store the contribution to the pooled covariance matrix
    poolCov = poolCov+ssd;
  end
end
% Normalize the pooled covariance matrix: True number of examples that
% belong to any class, subtract the true number of non-empty classes
poolCov = poolCov./(sum(perClass)-nnz(perClass));
poolCov = poolCov./(sum(perClass)-1);

% Compute inverse regularized covariance matrices. Do this here, so that the
% apply phase is shorter!
C.invRegCov = zeros(dim, dim, nClasses);
C.logDetRegCov = zeros(1, nClasses);
for k = 1:nClasses,
  % Avoid empty classes
  if perClass(k)>0,
    % Regularize the within-class covariance matrix. First, add a bit of the
    % pooled covariance matrix to make the within-class covariance estimate
    % more stable
    regCov = (1-C.lambda)*C.classProb(k)*classCov(:,:,k) + C.lambda*poolCov;
    regCov = regCov./((1-C.lambda)*C.classProb(k)+C.lambda);
    % Next regularization: Add a bit of identity matrix
    traceCov = trace(regCov);
    regCov = (1-C.gamma)*regCov+C.gamma*traceCov/dim*speye(dim);
    % Use pinv to avoid numerical problems. slower than inv, but OK here
    C.invRegCov(:,:,k) = pinv(regCov);
    C.logDetRegCov(k) = log(det(regCov));
  else
    C.logDetRegCov(k) = -Inf;
  end
end

C.bias = opt.bias;
C.func = opt.func;