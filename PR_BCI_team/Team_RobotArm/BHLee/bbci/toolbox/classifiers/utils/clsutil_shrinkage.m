function [Cstar, gamma, T] = clsutil_shrinkage(X, varargin)
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

% 12-08-2010: adapted fast computation procedure for all four target
%             matrices: schultze-kraft@tu-berlin.de

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
                  'target', 'B', ...
                  'gamma', 'auto', ...
                  'verbose', 0);

if isequal(opt.gamma, 'auto'),
  gamma = NaN;
elseif isreal(opt.gamma),
  gamma = opt.gamma;
else
  error('value for OPT.gamma not understood');
end
  
%%% Empirical covariance
[p, n] = size(X);
Xn     = X - repmat(mean(X,2), [1 n]);
S      = Xn*Xn';
Xn2    = Xn.^2;

%%% Define target matrix for shrinkage
idxdiag    = 1:p+1:p*p;
idxnondiag = setdiff(1:p*p, idxdiag);
switch(upper(opt.target))
    case 'A'
        T = eye(p,p);
    case 'B'
        nu = mean(S(idxdiag));
        T  = nu*eye(p,p);
    case 'C'
        nu = mean(S(idxdiag));
        c  = mean(S(idxnondiag));
        T  = c*ones(p,p) + (nu-c)*eye(p,p);
    case 'D'
        T = diag(S(idxdiag));
    otherwise
        error('unknown value for OPT.target')
end

%%% Calculate optimal gamma for given target matrix
if isnan(gamma)
    
    V     = 1/(n-1) * (Xn2 * Xn2' - S.^2/n);
    gamma = n * sum(sum(V)) / sum(sum((S - T).^2));
    
    %%% Handle special cases
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
    
end

%%% Estimate covariance matrix
Cstar = (gamma*T + (1-gamma)*S ) / (n-1);
