function [fv, opt, feat_score, sortedEig, V]= proc_pr_pca(fv, retain, varargin)
% proc_pr_pca - Principal component analysis (PCA) of features and data centering
%
% Synopsis:
%   fv = proc_pr_pca(fv)
%   [fv,opt,feat_score,D,V] = proc_pr_pca(fv,retain,'Property',Value,...)
%   
% Arguments:
%  fv: Feature vector structure. Only required field is fv.x (size 
%      [dim N] for N data points). Alternatively, fv can also be a plain 
%      [dim N] data matrix.
%  retain: Scalar. threshold for determining how many features to retain,
%      meaning depends on opt.policy. Default value: 2
%   
% Returns:
%  fv: Struct. Feature vector with transformed data fv.x, transformation
%      is P*fv.x - b. If input argument is a [dim N] matrix, output fv is
%      also a plain data matrix of size [retained N]
%  opt: Options structure with input options, plus the parameters of the
%      data transformation. Most important fields:
%      opt.P: [retained dim] matrix. Transformation matrix for data, P has the
%          leading eigenvectors as rows
%      opt.b: [retained 1] vector. Offset for data centering
%      Output data is opt.P*fv.x - b*ones([1 size(fv.x,2)])
%  feat_score: [retained 1] vector. Variance (eigenvalues) of the
%      retained components
%  D: [dim 1] vector. All eigenvalues sorted in descending order
%  V: [dim dim] matrix. Eigenvectors of the data covariance matrix, one
%      eigenvector per column, sorted by eigenvalues in descending order
%   
% Properties:
%  policy: String, one of 'number_of_features', 'perc_of_features',
%      'perc_of_score'. Strategy to choose the number of retained
%      features. Default: 'number_of_features'
%  whitening: Logical. 0: no whitening (default), P is orthonormal. 1:
%      whitening, P is not orthonormal
%  dim: Dimension that is to be projected, default 1. The non-default
%      choice of dim=2 is implemented in a dirty manner, time and
%      memory consuming.
%  P: [retained dim] matrix. Transformation matrix for data. If this
%      option is given, no new PCA is computed. Input fv is transformed
%      according to transformation given by b and P options.
%  b: [retained 1] vector. Offset for data centering
%   
% Description:
%   The usual principal component analysis. Policy 'perc_of_score'
%   chooses as many PCA components as required to explain the given
%   percentage of data variance.
%   
% Examples:
%   PCA for visualization (use the default of retaining 2 PCA components)
%     [fv2,opt] = proc_pr_pca(fv);
%     plot(fv2.x(1,:), fv2.x(2,:), 'k.');
%   Plot new data in the same coordinate system
%     B = opt.P*xnew-opt.b*ones([1 size(xnew,2)]);
%     plot(B(1,:), B(2,:), 'r.');
%   Do the above with using return argument opt:
%     [fv2,opt] = proc_pr_pca(fv, 2);
%     B = proc_pr_pca(xnew, [], opt);
%
%   
% See also: eig, eigs
% 

% Author(s), Copyright: 
% fcm 16jul2004
% fcm 08apr2005: changed all fv.x to fv.x' to meet the toolbox standard.
% Anton Schwaighofer, Jul 2005
% Benjamin Blankertz, Jun 2006: option dim=2
% $Id: proc_pr_pca.m,v 1.4 2006/06/08 08:17:55 neuro_toolbox Exp $

if nargin<2,
  retain = [];
end
if isempty(retain),
  retain = 2;
end
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'policy', 'number_of_features', ...
                        'whitening', 0, ...
                        'P', [], ...
                        'b', [], ...
                        'dim', 1);

if opt.dim==2,
  %% poor boy ...
  if isstruct(fv),
    fv.x= fv.x';
  else
    fv= fv';
  end
  opt.dim= 1;
  if ~isempty(opt.P),
    %% we need a special case for this, as we only may have 2 output args
    opt.P= opt.P';
    opt.b= opt.b';
    [fv, opt]= proc_pr_pca(fv, retain, opt);
  else
    [fv, opt, feat_score, sortedEig, V]= proc_pr_pca(fv, retain, opt);
    V= V';
  end
  opt.dim= 2;
  opt.P= opt.P';
  opt.b= opt.b';
  if isstruct(fv),
    fv.x= fv.x';
  else
    fv= fv';
  end
  return;
end

if isstruct(fv),
  x0 = fv.x;
else
  x0 = fv;
end
if ndims(x0)>2,
  error('proc_pr_pca can only be used with 2-dimensional data');
end
[dim N] = size(x0);
% Transformation matrices are given: apply them to the data, then return
if ~isempty(opt.P) & ~isempty(opt.b),
  if nargout>3,
    error('With matrix P and b given, only 3 output arguments are allowed');
  end
  x1 = opt.P*x0 - opt.b*ones([1 N]);
  if isstruct(fv),
    % Struct in, struct out:
    fv.x = x1;
  else
    % matrix in, matrix out:
    fv = x1;
  end
  return;
elseif xor(isempty(opt.P), isempty(opt.b)),
  error('Both options ''P'' and ''b'' must be supplied');
end

  
% ensure that the number of features to keep is smaller or equal
% to the actual dimension of the data set
switch opt.policy,
 case 'number_of_features',
  retain = min([retain, dim]);
  n_keep = retain;
 case 'perc_of_features',
  retain = min([retain, 100]);
  n_keep = max([round(retain/100*N), 1]);
 case 'perc_of_score',
  retain = min([retain, 100]);
end;

% centering the data featurewise
b0 = mean(x0,2);
x = x0 - b0 * ones(1,N); 

% calculate the eigenvalues of the covariance matrix. We could compute
% all eigenvectors in all cases, but this is a waste of time for the
% policies perc_of_features and number_of_features
C = (x*x')/(N-1);
switch opt.policy,
  case {'number_of_features', 'perc_of_features'}
    if nargout>3,
      % If V and D should also be returned, we can not use eigs
      [V D] = eig(C);
    else
      % Turn off eigs diagnostic message about Ritz values
      eigsOpts = struct('disp', 0);
      [V, D] = eigs(C, n_keep, 'LA', eigsOpts);
    end
    [sortedEig, ind] = sort(-diag(D));
  case 'perc_of_score'
    % Compute all eigenvalues:
    [V D] = eig(C);
    % Sort in descending order
    [sortedEig, ind] = sort(-diag(D));
    % Compute variance explained by each component. The minus introduced
    % when sorting cancels out here
    q = cumsum(sortedEig/sum(sortedEig));
    n_keep = min(find(q > retain/100));
end
% Retain only the eigenvectors with highest eigenvalues
P = V(:, ind(1:n_keep))';
sortedEig = -sortedEig;
feat_score = sortedEig(1:n_keep);
x1 = P*x;
if opt.whitening,
  W = spdiag( 1./sqrt( sum(x1.*x1,2)/(N-1) ) );
  x1 = W*x1;
  P = W*P;
end;
if isstruct(fv),
  % Struct in, struct out:
  fv.x = x1;
else
  % matrix in, matrix out:
  fv = x1;
end
b = P*b0;
opt.P = P;
opt.b = b;
