function [out,out_orig] = apply_RDAdist(C, X)
% APPLY_RDADIST - Apply RDA classifier with reject option by mahalonobis dist
%
%   Regularized Discriminant Analysis, with ability to reject examples if
%   their log-likelihood is too low.
%   
% Usage:
%   OUT = APPLY_RDADIST(C, X)
%   [OUT,OUT_NOREJECTS] = APPLY_RDADIST(C, X)
%   
% Input:
%   C: trained classifier, as output by TRAIN_RDADIST
%   X: Data matrix of test points, with one point per column (see
%      DATAFORMATS) 
%
% Output:
%   OUT: Matrix of test point log-likelihoods, this has size 
%        [NCLASSES NPOINTS]. For 2-class problems, log-odds are returned
%        in OUT of size [1 NPOINTS]. 
%        If a test point has been rejected, the corresponding column is
%        marked with NaNs.
%   OUT_NOREJECTS: Format as OUT, but without any NaN markers
%
%   
%   See also TRAIN_RDAREJECT,TRAIN_RLDA,DEMO_REJECT
%

% Copyright Anton Schwaighofer (2004)
% $Id: apply_RDAdist.m,v 1.2 2004/09/10 10:53:38 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

if ~isfield(C,'func')
  C.func = inline('sqrt(x)','x');
end

if ~isfield(C,'bias');
  C.bias = 0;
end


nClasses = length(C.classProb);
[nDim,nPoints] = size(X);
dim = size(X, 1);
% Output is a matrix containing the value for each of the discrimination
% functions
out = zeros([nClasses nPoints]);
dist = zeros([nClasses nPoints]);
for k = 1:nClasses,
  if C.classProb(k)>0,
    % Subtract class mean from all data points
    Xnorm = X-repmat(C.classMean(:,k), [1 nPoints]);
    % Compute the standard Gaussian log-likelihood (based on the
    % regularized covariance matrix)
    dist(k,:) = sum(Xnorm.*(C.invRegCov(:,:,k)*Xnorm), 1);
    out(k, :) = log(C.classProb(k)) - 0.5*C.logDetRegCov(k) - 0.5*dist(k,:);
  else
    out(k,:) = -Inf;
  end
end


if nargout>1,
  out_orig = out;
end

if strcmp(lower(C.rejectBelow), 'fadeout'),
  % Use the smooth fading out approximation instead of NaN markers
  if nClasses==2,
    % singleOut>0 indicates class 2, otherwise class 1
    singleOut = out(2,:)-out(1,:)+C.bias;
  else
    error('The ''fadeOut'' option can only be used for 2-class problems');
  end
  % Index of classifier output for the winning class, for each point:
  winnerInd = (1:2:(2*nPoints))+(singleOut>0);
  % Multiply classifier output with the class-conditional density of the
  % winning class. Classifier output is high near the class centres, with
  % a smooth transition between the classes along the decision boundary.
%  out = singleOut.*exp(out(winnerInd)/C.damp)./(1+exp(out(winnerInd)/C.damp));
  out = nDim*singleOut./feval(C.func,dist(winnerInd));
else
  % Rejecting with NaN markers: If *all* classes indicate a very low
  % log-likelihood, we reject. Still, it is unclear whether this should be
  % based on the values in OUT (that include the class probabilities) or on
  % the pure Gaussian likelihoods
  reject = all(out<C.rejectBelow, 1);
  % Mark them with NaNs
  out(:,reject) = NaN;
  % Simplify for 2-class case: out>0 indicates class 2, otherwise class 1
  if nClasses==2,
    out = out(2,:)-out(1,:);
  end
end
