function C = train_RDAreject(X, labels, varargin)
% train_RDAreject - Train RDA classifier, with reject option at apply
%
% Synopsis:
%   C = train_RDAreject(X,LABELS)
%   C = train_RDAreject(X,LABELS,'Property',Value,...)
%   
% Arguments:
%   X: [d,N] matrix of input data, with one point/sample per column
%   LABELS: 0/1 matrix indicating class membership. See DATAFORMATS for
%      details 
%   
% Returns:
%   C: The trained classifier. C is a structure containing the following 
%      fields:
%      .classProb: Assumed prior class probabilities
%      .classMean: [d C] matrix containing the class mean vector each of
%         of the C classes.
%      .rejectBelow, .lambda', .gamma: Copies of the input propertiers
%      .invRegCov: [d d C] matrix, with the inverse regularized
%         covariance matrix of class i in [:,:,i].
%      .logDetRegCov: [1 d] vector, with -log(det(C.invRegCov(:,:,i))) 
%         in the i.th element
%   
% Properties:
%   'lambda': regularization parameter LAMBDA for RDA. Morph between
%      QDA (LAMBDA=0) and LDA (LAMBDA=1). Default value: 1.
%   'gamma': regularization parameter GAMMA for RDA. Morph between
%      covariance matrix estimated from data (GAMMA=0) and spherical
%      covariance matrix (GAMMA=1).
%   'classProb': Use the values passed here as the a-priori class
%      probabilities. If omitted, the class probabilities will be computed
%      from the training data. 
%   'rejectMethod': One of 'threshold', 'classCond', 'custom'.
%   'rejectParam': Parameter for rejection, interpretation depends on
%      chosen 'rejectMethod'.
%   'bias': Bias term for the classifier output, used only for 2-class
%      problems. Default value: 0
%   Default values: 'rejectMethod'=='threshold', 'rejectParam'==[-Inf]
%      (no rejections at all)
%
% Description:
%   Regularized Discriminant Analysis, with ability to reject examples,
%   based on different heuristics. Method for rejection is chosen by the
%   'rejectMethod' option:
%   'rejectMethod'=='threshold': Reject examples if the log-likelihood is
%      below a threshold value for all classes. Rejected examples are
%      marked by NaN in the classifier output. Rejection threshold is
%      given in option 'rejectParam' (scalar value)
%   'rejectMethod'=='classCond': Scale classifier outputs such that
%      rejected examples get a near-zero value.
%      This is done by multiplying the classifier output with the
%      (scaled) class-conditional density of the winning class. 
%      Scaling is based on two values given in 'rejectParam', 
%      'rejectParam'==[PROBMASS SCALING]:
%      - Compute the distance D from the mean such that the given
%        probability mass PROBMASS is covered by this region.
%      - Scale Mahalanobis distance such that 
%        exp(-0.5*D*factor)==SCALING (points at distance D from the mean
%        have output SCALING times original classifier output)
%   'rejectMethod'=='customScaling': Customized scaling. Classifier output is
%      multiplied by the output of a user-supplied scaling function
%      (function name/handle in 'rejectParam' option). This function gets
%      the squared Mahalanobis-distance as input, and returns a scaling
%      factor.
%   OPTS.shrinkage: if true, the shrinkage parameter is selected by the
%   function clsutil_shrinkage. See train_RLDAshrink for details.
%
% Examples:
%   C = train_RDAreject(X, Labels, 'classProb', [0.2 0.8], ...
%       'rejectMethod', 'threshold', 'rejectParam', -2)
%      trains the RDA with manually setting the prior class probabilities to 
%      [0.2 0.8]. Outliers that have a log-likelihood below -2 will be
%      marked as NaN.
%   C = train_RDAreject(X, Labels, 'rejectMethod', 'classCond', ...
%       'rejectParam', [0.95 0.001]);
%      trains a classifier that scales outliers to a near-zero
%      output. Points at class mean retain their original classifier
%      output value. Points at a distance that covers 95% of the
%      probability mass are scaled down by a factor of 0.001.
%   C = train_RDAreject(X, Labels, 'rejectMethod', 'custom', ...
%       'rejectParam', inline('x.^(-2)','x'));
%      will return a classifier that scales points far from the class
%      center by the inverse 4th power of the Mahalanobis distance.
%      Mind that this may be problematic in high dimensions. Use a higher
%      power of the scaling function as dimensionality increases.
%
% References:
%   J.H. Friedman, Regularized Discriminant Analysis, Journal
%   of the Americal Statistical Association, vol.84(405), 1989.
%   
% See also: train_RLDA,apply_RDAreject,train_RDA,demo_reject,inline
% 

% Author(s): Anton Schwaighofer, Sep 2004
% Shrinkage added by Benjamin and Michael, Dez. 2009
% $Id: train_RDAreject.m,v 1.10 2005/05/04 15:22:17 neuro_toolbox Exp $

error(nargchk(2, Inf, nargin));

opt = propertylist2struct(varargin{:});
% Set default parameters
opt = set_defaults(opt, 'lambda', 1, ...
                        'gamma', 0.01, ...
                        'rejectMethod', 'threshold', ...
                        'rejectParam', -Inf, ...
                        'bias', 0, ...
                        'shrinkage',0, ...
                        'classProb', []);
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

if ~any(strcmp(opt.rejectMethod, {'threshold', 'classCond', 'customScaling'})),
  error('Invalid value for option ''rejectMethod''');
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
C = opt;
if isfield(C, 'isPropertyStruct'),
  C = rmfield(C, 'isPropertyStruct');
end
C.type = 'RLDAreject';
% Extract the number of examples per class - easy since labels are now 0/1
perClass = sum(labels, 2);
% Use class probabilities from options or from training data
if ~isempty(opt.classProb),
  % Class probabilities from options: make sure there are examples for
  % all classes with non-zero probability
  missingExamples = (C.classProb>0) & (perClass==0);
  if any(missingExamples),
    error(sprintf('No examples given for class %i', find(missingExamples)));
  end
else
  % Estimate class probabilities from training data:
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
    % This gives the covariance matrix
    if opt.shrinkage,
      ssd = clsutil_shrinkage(Xnorm);
    else
        ssd = Xnorm*Xnorm';
    end
    classCov(:,:,k) = ssd./(perClass(k)-1);
    % Also store the contribution to the pooled covariance matrix
    poolCov = poolCov+ssd;
  end
end
% Normalize the pooled covariance matrix: True number of examples that
% belong to any class, subtract the true number of non-empty classes
poolCov = poolCov./(sum(perClass)-nnz(perClass));

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
    % Sometimes covariance matrices are really awful (thanks to Benjamin for
    % pointing me to this fact) and have det=Inf. Do not compute
    % log(det(regCov)) directly, but via Cholesky
    C.logDetRegCov(k) = 2*sum(log(diag(chol(regCov))));
  else
    C.logDetRegCov(k) = -Inf;
  end
end
