function C = train_mixGauss(xTr,yTr,k,la,ga,varargin)
%TRAIN_MIXGAUSS calculates mixtures of Gaussians to the
%data for classification
%
% usage:
%    C = train_mixGauss(xTr,yTr,k,la,ga,epsilon,iter)
%
% input:
%    xTr     the data
%    yTr     the labels
%    k       the number of Clusters for each class, can be one
%            value, then it is the same value for each class
%    la      see RDA
%    ga      see RDA
%    for the other parameters see mixog
%
% output
%    C       a Classifer which can be used by
%            apply_mixturesofGaussian2
%
% based on Roweis, Gharamani
% Guido Dornhege, 08/04/03

if ~exist('la','var') | isempty(la)
  la = 0;
end

if ~exist('ga','var') | isempty(ga)
  ga = 0;
end

if ~exist('k') | isempty(k)
  C = train_LDA(xTr,yTr);
  C.k = 1;
  return
end

if size(yTr,1)==1
  yTr = [yTr<0;yTr>0];
end

nClasses = size(yTr,1);

if length(k)==1
  k = k*ones(1,nClasses);
end

nDim = size(xTr,1);

means = zeros(nDim,sum(k));
covs = zeros(nDim,nDim,sum(k));
priors = zeros(1,sum(k));
detcovs = zeros(1,sum(k));
st = [0,cumsum(k)];

for i = 1:nClasses
  x = xTr(:,find(yTr(i,:)));
  [me,co,pr] = mixog(x,k(i),varargin{:});
  ldco = det(co);
  means(:,st(i)+1:st(i+1)) = me;
  covs(:,:,st(i)+1:st(i+1)) = repmat(co,[1 1 k(i)]);
  priors(st(i)+1:st(i+1)) = pr;
  detcovs(st(i)+1:st(i+1)) = ldco*ones(1,k(i));
end

if (la~=0)
  covges = sum(covs.*repmat(reshape(priors,[1 1 sum(k)]),[nDim,nDim,1]),3);
  covs = (1-la)*covs + la*repmat(covges,[1 1 sum(k)]);
end

if (ga~=0)
  for i = 1:size(covs,3)
    covs(:,:,i) = (1-ga)*covs(:,:,i) + ga/nDim*trace(covs(:,:,i))* ...
	eye(nDim);
  end
end

for i = 1:size(covs,3)
  covs(:,:,i) = pinv(covs(:,:,i));
end

  
C.means = means;
C.covs = covs;
C.priors = priors;
C.detcovs = detcovs;
C.k = k;
