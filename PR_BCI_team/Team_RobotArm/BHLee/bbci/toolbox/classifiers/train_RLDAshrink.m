function C = train_RLDAshrink(xTr, yTr, varargin)
% TRAIN_RLDASHRINK - Regularized LDA with automatic shrinkage selection
%
% Usage:
%   C = train_RLDAshrink(X, LABELS)
%   C = train_RLDAshrink(X, LABELS, OPTS)
%
% Input:
%   X: Data matrix, with one point/example per column. 
%   LABELS: Class membership. LABELS(i,j)==1 if example j belongs to
%           class i.
%   OPTS: options structure as output by PROPERTYLIST2STRUCT. Options
%           are also passed to clsutil_shrinkage.
%     .scaling: scale projection vector such that the distance between
%        the projected means becomes 2
%     .store_means: if true, the classwise means of the feature vectors
%        are stored in the classifier structure C. This can be used, e.g.,
%        for bbci_adaptation_pmean
%     .store_invcov: if true, the extended covariance matrix is stored in
%        the classifier structure C. This can be used, e.g., for
%        bbci_adaptation_pcovmean
%     .use_pcov: if true, the pooled covariance matrix is used instead
%        of the average classwise covariance matrix.
%
% Output:
%   C: Classifier structure, hyperplane given by fields C.w and C.b.
%      Optionally the classwise means are stored in C.means and the
%      inverse of the covariance matrix is stored in C.invcov.
%
% Description:
%   TRAIN_RLDA trains a regularized LDA classifier on data X with class
%   labels given in LABELS. The shrinkage parameter is selected by the
%   function clsutil_shrinkage.
%
%   References: J.H. Friedman, Regularized Discriminant Analysis, Journal
%   of the Americal Statistical Association, vol.84(405), 1989. The
%   method implemented here is Friedman's method with LAMBDA==1. The
%   original RDA method is implemented in TRAIN_RDAREJECT.
%
% Example:
%   train_RLDA(X, labels)
%   train_RLDA(X, labels, 'target', 'D')
%   
%See also APPLY_SEPARATINGHYPERPLANE, CLSUTIL_SHRINKAGE, 
%   TRAIN_LDA, TRAIN_RDAREJECT

% blanker@cs.tu-berlin.de

if size(yTr,1)==1, yTr= [yTr<0; yTr>0]; end
nClasses= size(yTr,1);

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'exclude_infs', 0, ...
                 'prior', ones(nClasses, 1)/nClasses, ...
                 'use_pcov', 0, ...
                 'store_prior', 0, ...
                 'store_means', 0, ...
                 'store_cov', 0, ...
                 'store_invcov', 0, ...
		 'store_extinvcov',0, ...
                 'scaling', 0);

% empirical class priors as an option (I leave 1/nClasses as default, haufe)
if isnan(opt.prior)
  opt.prior = sum(yTr, 2)/sum(sum(yTr));
end

if opt.exclude_infs,
  ind = find(sum(abs(xTr),1)==inf);
  xTr(:,ind) = [];
  yTr(:,ind) = [];
end

d= size(xTr, 1);
X= zeros(d,0);
C_mean= zeros(d, nClasses);
for ci= 1:nClasses,
  idx= find(yTr(ci,:));
  C_mean(:,ci)= mean(xTr(:,idx),2);
  if ~opt.use_pcov,
    X= [X, xTr(:,idx) - C_mean(:,ci)*ones(1,length(idx))];
  end
end
if opt.use_pcov,
  [C_cov, C.gamma]= clsutil_shrinkage(xTr, opt);
else
  [C_cov, C.gamma]= clsutil_shrinkage(X, opt);
end
C_invcov= pinv(C_cov);

C.w= C_invcov*C_mean;
C.b= -0.5*sum(C_mean.*C.w,1)' + log(opt.prior);

if nClasses==2
  C.w= C.w(:,2) - C.w(:,1);
  C.b= C.b(2)-C.b(1);
end

if opt.scaling,
  if nClasses>2,
    error('scaling only implemented for 2 classes so far (TODO!)');
  end
  if ~isdefault.prior,
    warning('prior ignored, when scaling (TODO!)');
  end
  C.w= C.w/(C.w'*diff(C_mean, 1, 2))*2;
  C.b= -C.w' * mean(C_mean,2);
end

if opt.store_prior,
  C.prior= opt.prior;
end
if opt.store_means,
  C.mean= C_mean;
end
if opt.store_cov,
  C.cov= C_cov;
end
if opt.store_invcov,
  C.invcov= C_invcov;
end
if opt.store_extinvcov,
  % pooled(!) covariance
  feat= [ones(1,size(xTr,2)); xTr];
  C.extinvcov= inv(feat*feat'/size(xTr,2));
  % Alternative (with shrinkage):
%  [C_extpcov, C_gamma_extpcov]= clsutil_shrinkage([ones(1,size(xTr,2)); xTr]);
%  C.extinvcov= pinv(C_extpcov);
% But this subtracts the pooled mean, which seems not to be appropriate
% ?? Ask Carmen, when she's back ??
end
