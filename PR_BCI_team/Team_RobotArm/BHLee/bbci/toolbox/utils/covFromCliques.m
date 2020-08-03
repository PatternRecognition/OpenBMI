function [covMat,mu,concMat] = covFromCliques(X, cliques, varargin)
% COVFROMCLIQUES - Estimate covariance of Gaussian model with given cliques
%
% Estimate covariance matrix of a Gaussian graphical model, assuming
% either a decomposable or non-decomposable model structure
%
% Usage:
%   [K,MU,C] = COVFROMCLIQUES(X, CLIQUES, <OPTS>)
%     (assuming a non-decomposable model, implemented with Iterative
%     Proportional Scaling)
%   [K,MU,C] = COVFROMCLIQUES(X, CLIQUES, SEPARATORS, <OPTS>)
%     (assuming a decomposable model with given cliques and
%     separators. Exact equations available)
%
% Input:
%   X: Data matrix, with one data point per column. For N points in D
%      dimensions, X has size [D N].
%   CLIQUES: A cell array of length C for C cliques, containing in
%       CLIQUES{i} the dimensions that make up the i.th
%       clique. Alternatively, CLIQUES{i} can be a logical index vector
%       of length D.
%   SEPARATORS: A cell array of length S for S separators. Similar to
%       CLIQUES, this can either contain index vectors or logical
%       vectors.
%   Mind that only minimal consistency checking is done. In particular, it
%   is not checked whether CLIQUES and SEPARATORS form a valid decomposable
%   model.
%   OPTS: Property/value list of options. Recognized options:
%     'maxSteps': Maximum number of iterations for Iterative Proportional
%       scaling. Default value: 10
%     'belowNorm': Terminate if norm of the covariance update is below
%       the value given here. Default: 1e-6
%     'whichNorm': Defines the matrix norm that is used for checking
%       convergence. The value here may be any value accepted by NORM.
%     'verbosity': Values 0, 1 or 2 print more and more progress
%       information. 'verbosity'=3 plots the matrices in each step.
%
% Output:
%   K: Estimated covariance matrix
%   MU: Estimated mean vector.
%   C: Estimated concentration matrix (inverse covariance matrix)
%
%   References: Lauritzen: Graphical Models. Oxford University Press,
%   1996, sec. 5.2 (for Gaussian models) and sec. 5.3 (decomposable
%   Gaussian models).
%   
%   See also COV,SPARSEBLOCK,NORM
%

% Copyright (c) by Anton Schwaighofer (2004)
% $Id: covFromCliques.m,v 1.4 2004/11/03 13:43:05 neuro_cvs Exp $

error(nargchk(2, Inf, nargin));

if nargin<3,
  % No separators given: Assume general model, fit by IPS
  separators = {};
  decompModel = 0;
  opts = [];
else
  if iscell(varargin{1}),
    % Cell as third argument: Should be separators. Assume decomposable model,
    % fit by exact equations
    separators = varargin{1};
    decompModel = 1;
    % Options start right after
    opts = propertylist2struct(varargin{2:end});
  else
    % Third arg is not a cell: Assume these are options
    separators = {};
    decompModel = 0;
    opts = propertylist2struct(varargin{:});
  end
end

opts = set_defaults(opts, 'maxSteps', 10, ...
                          'belowNorm', 1e-6, ...
                          'whichNorm', 2, ...
                          'verbosity', 1);

[dim nPoints] = size(X);
% Make a bit of consistency check for the cliques and separators:
nCliques = length(cliques);
nSeps = length(separators);
% Union of all variables appearing in cliques
allVars = logical(zeros([1 dim]));
% These are the variables (dimensions) that may be contained in cliques
validVars = 1:dim;
for i = 1:nCliques,
  C = cliques{i};
  if islogical(C),
    if ~length(C)==dim,
      error(sprintf('Clique %i must be a logical vector of D (for D dimensions)', i));
    end
  elseif ~isempty(setdiff(C, validVars)),
    error(sprintf('Clique %i contains invalid indices', i));
  end
  allVars = union2(allVars, C);
end
% All variables must be contained in at least one clique. allVars should
% be a logical vector now, containing ones
if length(allVars)~=dim | ~all(allVars),
  error('Set of cliques is invalid');
end
% Check for separators: Only can check whether the indices are correct,
% not whether separators and cliques match
for i = 1:nSeps,
  S = separators{i};
  if islogical(S),
    if ~length(S)==dim,
      error(sprintf('Separator %i must be a logical vector of D (for D dimensions)', i));
    end
  elseif ~isempty(setdiff(S, validVars)),
    error(sprintf('Separator %i contains invalid indices', i));
  end
end

% Mean estimate is independent of the model's graph structure.
mu = mean(X, 2);
% Subtract mean for all further computations
X = X-repmat(mu, [1 nPoints]);
% For iterative proportional scaling, we often need the sum-of-squares
% matrices (unscaled covariance estimates) for each clique. Precompute
% those (Lauritzens 'ssd' variables for cliques and separators)
ssdCliques = cell([1 nCliques]);
for i = 1:nCliques,
  % Extract those dimensions that correspond to the current clique
  Xc = X(cliques{i},:);
  ssdCliques{i} = Xc*Xc';
end
ssdSeps = cell([1 nSeps]);
for i = 1:nSeps,
  % Extract those dimensions that correspond to the current separator
  Xs = X(separators{i},:);
  ssdSeps{i} = Xs*Xs';
end

if ~decompModel,
  % Called with 2 input arguments: Assume that we have a general undirected
  % Gaussian model. Fit covariance matrix, or rather, concentration matrix
  % C, by iterative proportional scaling (IPS). Equations are taken from
  % Lauritzen, sec. 5.2 (equ. 5.16 and 5.20)

  % Starting value for concentration matrix: inverse variances. Starting
  % with a submatrix of the full covariance leads to convergence problems
  varX = sum(X.*X, 2)./nPoints;
  concMat = diag(1./varX);
  % For IPS, we need the covariance matrix in each step as well
  % (Lauritzen's K^{-1}_CC)
  covMat = diag(varX);

  r = 1;
  normReached = 0;
  while r<=opts.maxSteps,
    % Log the maximum norm of the update matrices encountered within each
    % IPS step
    maxNorm = -Inf;
    for i = 1:nCliques,
      C = cliques{i};
      % Lauritzen, equ. 5.16: 'adjusting the C marginal'
      updateMat = nPoints*inv(ssdCliques{i}) - inv(covMat(C,C));
      concMat = concMat + sparseblock(C, C, updateMat, dim, dim);
      % Immediately update the covariance matrix. Lauritzen is a bit
      % unclear in equ.5.16, in when to do this update. For large
      % problems, updating covMat after all clique marginals had been
      % updated did not converge.
      covMat = inv(concMat);
      % Log norm
      updateNorm = norm(updateMat, opts.whichNorm);
      maxNorm = max(maxNorm, updateNorm);
    end
    % Updating covMat here does not converge (only for small toy
    % problems, not for the large BCI data). Thus: update in the loop above!
    % covMat = inv(concMat);
    if opts.verbosity>1,
      fprintf('IPS step %i: Update with maximum norm of %g\n', r, maxNorm);
    end
    if opts.verbosity>2,
      figure;
      subplot(1, 2, 1);
      imagesc(covMat);
      colorbar;
      title(sprintf('Covariance matrix after %i IPS steps', r));
      subplot(1, 2, 2);
      imagesc(concMat);
      colorbar;
      title(sprintf('Concentration matrix after %i IPS steps', r));
    end
    % Check for termination
    if maxNorm<opts.belowNorm,
      normReached = 1;
      break;
    end
    r = r+1;
  end
  if opts.verbosity>0,
    if normReached,
      fprintf('IPS terminated successfully.\n');
    else
      fprintf('IPS: Maximum number of steps reached.\n');
    end
    fprintf('Update in last iteration with maximum norm of %g, required was %g.\n', ...
            maxNorm, opts.belowNorm);
  end
else
  % Separators given: Assume that we have a decomposable model. Thus,
  % exact equations are available, no iterative fitting
  % necessary. Equations are from Lauritzen, Proposition 5.9
  concMat = zeros([dim dim]);
  for i = 1:nCliques,
    C = cliques{i};
    % Take the ssd for the current clique, invert and expand to full
    % matrix size (Lauritzen's ^\Gamma operation, implemented with
    % SPARSEBLOCK here)
    concMat = concMat + sparseblock(C, C, inv(ssdCliques{i}), dim, dim);
  end
  % Subtract the contribution of the separators
  for i = 1:nSeps,
    S = separators{i};
    concMat = concMat - sparseblock(S, S, inv(ssdSeps{i}), dim, dim);
  end
  % That's it... rescale and we are done!
  concMat = nPoints*concMat;
  covMat = inv(concMat);
end
