function C = train_RLDA(xTr, yTr, gamma)
% TRAIN_RLDA - Train regularized linear discriminant analysis (RLDA)
%
% Usage:
%   C = TRAIN_RLDA(X, LABELS, GAMMA)
%   C = TRAIN_RLDA(X, LABELS, OPTS)
%
% Input:
%   X: Data matrix, with one point/example per column. 
%   LABELS: Class membership. LABELS(i,j)==1 if example j belongs to
%           class i.
%   GAMMA: RLDA regularization parameter, with values between 0 and 1.
%          GAMMA=0 gives normal LDA, GAMMA=1 uses a multiple of the
%          identity matrix instead of the pooled covariance matrix.
%   OPTS: options structure as output by PROPERTYLIST2STRUCT. Recognized
%         options are:
%         'gamma': regularization parameter GAMMA, see above.
% Output:
%   C: Classifier structure, hyperplane given by fields C.w and C.b
%
% Description:
%   TRAIN_RLDA trains a regularized LDA classifier on data X with class
%   labels given in LABELS. GAMMA is the RLDA regularization parameter, with
%   values between 0 and 1. With GAMMA=0, the estimated pooled covariance
%   matrices will be used in LDA. GAMMA>0 adds small multiples of the
%   identity matrix to the covariance, up to the limit GAMMA=1 where the
%   pooled covariance matrix is assumed to be a (multiple of the) identity
%   matrix.
%
%   References: J.H. Friedman, Regularized Discriminant Analysis, Journal
%   of the Americal Statistical Association, vol.84(405), 1989. The
%   method implemented here is Friedman's method with LAMDBA==1. The
%   original RDA method is implemented in TRAIN_RDAREJECT.
%
% Example:
%   train_RLDA(X, labels, 0.2)
%   train_RLDA(X, labels, propertylist2struct('gamma', 0.2))
%   
%   
%   See also APPLY_SEPARATINGHYPERPLANE,TRAIN_LDA,TRAIN_RDAREJECT
%

% Copyright Fraunhofer FIRST.IDA (2004)
% $Id$

% Standard input argument checking
error(nargchk(2, 3, nargin));

% No, though shallst not use 'exist' for variables
if nargin<3 | isempty(gamma),
  gamma = 0;
end
% Now comes the new part, where the options can also be passed as an options
% structure. Check whether GAMMA has been created by PROPERTYLIST2STRUCT:
if ispropertystruct(gamma),
  if nargin>3,
    error('With given OPTS, no additional input parameter is allowed');
  end
  % OK, so the third arg was not gamma, but the options structure
  opt = gamma;
  % Set default parameters
  opt = set_defaults(opt, 'gamma', 0);
  % Extract parameters from options
  gamma = opt.gamma;
end

if gamma<0 | gamma>1,
  error('Regularization parameter GAMMA must be between 0 and 1');
end

% Here starts the original code of train_RLDA. Untouched apart from
% renaming ga by gamma
if size(yTr,1) == 1 yTr = [yTr<0; yTr>0]; end

ind = find(sum(abs(xTr),1)==inf);
xTr(:,ind) = [];
yTr(:,ind) = [];

nClasses= size(yTr,1);
clInd= cell(nClasses,1);
N= zeros(nClasses, 1);
for ci= 1:nClasses,
  clInd{ci}= find(yTr(ci,:));
  N(ci)= length(clInd{ci});
end

priorP = ones(nClasses,1)/nClasses;   

d= size(xTr,1);
m= zeros(d, nClasses);
Sq= zeros(d, d);
for ci= 1:nClasses,
  cli= clInd{ci};
  m(:,ci)= mean(xTr(:,cli),2);
  yc= xTr(:,cli) - m(:,ci)*ones(1,N(ci));
  Sq= Sq + yc*yc';
end
Sq= Sq/(sum(N)-1);
Sq = (1-gamma)*Sq + gamma/d*trace(Sq)*eye(d);
Sq = pinv(Sq);

C.w = Sq*m;
C.b = -0.5*sum(m.*C.w,1)' + log(priorP);

if nClasses==2
  C.w = C.w(:,2) - C.w(:,1);
  C.b = C.b(2)-C.b(1);
end
