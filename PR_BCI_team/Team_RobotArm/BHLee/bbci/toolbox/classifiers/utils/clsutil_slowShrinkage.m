function [Cstar, gamma, T]= clsutil_slowShrinkage(X, varargin)
%CLSUTIL_SHRINKAGE - Shrinkage of Covariance Matrix with 'Optimal' Parameter
%
%Synopsis:
%  [CSTAR, GAMMA, T]= clsutil_shrinkage(X, OPT);
%
%Arguments:
%  X: Data [nDim nSamples]
%  OPT: Struct or property/value list of optional properties
%   .target: target mode of shrinkage, cf. [2],
%      'A' (identity matrix),
%      'B' (diagonal, common variance): default
%      'C' (common covariance)
%      'D' (diagonal, unequal variance)
%   .gamma: shrinkage parameter. May be used to set the shrinkage
%      parameter explicitly.
%
%Returns:
%  CSTAR: estimated covariance matrix
%  GAMMA: selected shrinkage parameter
%  T: target matrix for shrinkage
%
% [1] Ledoit O. and Wolf M. (2004) "A well-conditioned estimator for
% large-dimensional covariance matrices", Journal of Multivariate
% Analysis, Volume 88, Number 2, February 2004 , pp. 365-411(47)
%
% [2] Sch�fer, Juliane and Strimmer, Korbinian (2005) "A Shrinkage
% Approach to Large-Scale Covariance Matrix Estimation and
% Implications for Functional Genomics," Statistical Applications in
% Genetics and Molecular Biology: Vol. 4 : Iss. 1, Article 32.

% suboptimal coding: blanker@cs.tu-berlin.de


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'target', 'D', ...
                  'gamma', 'auto', ...
                  'verbose', 0);

if isequal(opt.gamma, 'auto'),
  gamma= NaN;
elseif isreal(opt.gamma),
  gamma= opt.gamma;
else
  error('value for OPT.gamma not understood');
end
  
%% Empirical covariance
[p, n]= size(X);
Xn= X - repmat(mean(X,2), [1 n]);
Cemp= (Xn*Xn') / (n-1);

%% Define target matrix for shrinkage
idxdiag= 1:p+1:p*p;
idxnondiag= setdiff(1:p*p, idxdiag);
switch(upper(opt.target)),
 case 'A',
  T= eye(p, p);
 case 'B',
  nu= mean(Cemp(idxdiag));
  T= nu * eye(p,p);
 case 'C',
  nu= mean(Cemp(idxdiag));
  c= mean(Cemp(idxnondiag));
  T= c*ones(p,p) + (nu-c)*eye(p,p);
 case 'D',
  T= diag(diag(Cemp));
 otherwise,
  error('unknown value for OPT.target');
end

%% If gamma was specified explicitly, we can stop here.
if ~isnan(gamma),
  Cstar= gamma*T + (1-gamma)*Cemp;
  return;
end


%% Calculate optimal gamma for given target matrix
SumVarCii= 0;
for ii= 1:p,
  VarCii= var(Xn(ii,:).*Xn(ii,:));
  SumVarCii= SumVarCii + VarCii;
end
SumVarCii= n/(n-1)^2 * SumVarCii;
SumVarCij= 0;
if p>1,
  for ii= 1:p-1,
    for jj= ii+1:p,
      VarCij= var(Xn(ii,:).*Xn(jj,:));
      SumVarCij= SumVarCij + VarCij;
    end
  end
  SumVarCij= n/(n-1)^2 * 2*SumVarCij;
end
switch(upper(opt.target)),
 case 'A',
  gamma= (SumVarCij + SumVarCii) / sum(sum((Cemp-T).^2)); 
  
 case 'B',
  gamma= (SumVarCij + SumVarCii) / sum(sum((Cemp-T).^2)); 
  
 case 'C',
  gamma= (SumVarCij + SumVarCii) / sum(sum((Cemp-T).^2)); 
  
 case 'D',
  gamma= SumVarCij / sum(Cemp(idxnondiag).^2);
end

%% Handle special cases
if gamma>1,
  if opt.verbose,
    warning('gamma forced to 1');
  end
  gamma= 1;
elseif gamma<0,
  if opt.verbose,
    warning('gamma forced to 0');
  end
  gamma= 0;
end

Cstar= gamma*T + (1-gamma)*Cemp;
