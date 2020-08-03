function [means,covs,priors] = mixog(x,k,reg,epsilon,iter)
%CALCULATES MIXTURES OF GAUSSIANS
%
% usage:
%      [means,covs,priors] = mixog(x,<k,reg,epsilon,iter>);
%
% literature:
% A Unifying Review of Linear Gaussian Models, by Sam Roweis,
% Zoubin Gharamani
%
% input:
%      x              the data (written in column)
%      k              the number of gaussians (2)
%      reg            some small reg constant to be sure that
%                     matrices are positive definit. (1e-8)
%      epsilon        termination condition, change in log likelihood  (1e-5)
%      iter           termination condition, number of iterations (1000)      
%  
% output:
%      means          the means written in columns
%      covs           the over all covariance matrix 
%      priors         the priors as vector
%
% Guido Dornhege, 09/04/2003

if ~exist('iter','var') | isempty(iter)
  iter = 1000;
end

if ~exist('epsilon','var') | isempty(epsilon)
  epsilon = 1e-5;
end

if ~exist('reg','var') | isempty(reg)
  reg = 1e-8;
end


if ~exist('k','var') | isempty(k)
  k = 2;
end

[nDim,nTrials] = size(x);


%initialisation

ind = round(linspace(1,nTrials+1,k+1));
means = zeros(nDim,k);
covs = zeros(nDim,nDim);
for i = 1:k
  xx = x(:,ind(i):ind(i+1)-1);
  means(:,i) = mean(xx,2);
  da = xx-means(:,i)*ones(1,size(xx,2));
  covs = covs+da*da';
end

priors = ones(1,k)/k;

covs = covs/nTrials;
epsold = -inf;
Gamma = calcInferencemog(x,means,covs,priors);
epsi = sum(log(sum(Gamma,1)),2);

while abs(epsi-epsold)>epsilon & iter>0
  y = Gamma./(ones(k,1)*sum(Gamma,1));
  delta = x*y';
  gamma = sum(y,2);
  
  alpha = reg*eye(nDim,nDim);
  for j = 1:k
    means(:,j) = delta(:,j)/gamma(j);
    da = x-means(:,j)*ones(1,nTrials);
    alph = (ones(nDim,1)*y(j,:)).*da;
    alpha = alpha+alph*da';
  end

  covs = alpha/nTrials;
  priors = gamma/nTrials;
  epsold = epsi;
  Gamma = calcInferencemog(x,means,covs,priors);  
  epsi = sum(log(sum(Gamma,1)),2);
  iter = iter-1;
end










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Gamma = calcInferencemog(x,means,covs,priors);
%
% Gamma : nClusters*nTrials
nCluster = size(means,2);
nTrials = size(x,2);

Gamma = zeros(nCluster,nTrials);

for i = 1:nCluster
  da = x-means(:,i)*ones(1,nTrials);
  da = sum(da.*(pinv(covs)*da),1);
  Gamma(i,:) = priors(i)*exp(-0.5*da);
end



