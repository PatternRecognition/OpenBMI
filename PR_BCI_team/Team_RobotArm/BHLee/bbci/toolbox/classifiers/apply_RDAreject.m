function [out,out_orig] = apply_RDAreject(C, X)
% apply_RDAreject - Apply RDA classifier with reject option
%
% Synopsis:
%   [out,out_norejects] = apply_RDAreject(C, X)
%   
% Arguments:
%   C: trained classifier, as output by train_RDAreject
%   X: [ndim npoints] matrix of test points, with one point per column
%      (see dataformats.m) 
%
% Returns:
%   out: [nclasses npoints] matrix of test point log-likelihoods.
%        For 2-class problems, log-odds are returned, out has size 
%        [1 npoints] in this case
%        If a test point has been rejected, the corresponding column is
%        marked with NaNs.
%   out_norejects: Format as out, but without any NaN markers
%
% Description:
%   Regularized Discriminant Analysis, with ability to reject examples if
%   their log-likelihood is too low.
%
%   
%   See also train_RDAreject,train_RLDA,demo_reject
%

% Author(s): Anton Schwaighofer, Sep 2004
% $Id: apply_RDAreject.m,v 1.6 2004/10/27 16:31:41 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

nClasses = length(C.classProb);
nPoints = size(X, 2);
dim = size(X, 1);
% Output is a matrix containing the value for each of the discrimination
% functions
out = zeros([nClasses nPoints]);
% Mahalanobis-distance for test points in each class
mahalDist = zeros([nClasses nPoints]);

for k = 1:nClasses,
  if C.classProb(k)>0,
    % Subtract class mean from all data points
    Xnorm = X-repmat(C.classMean(:,k), [1 nPoints]);
    % (Squared) Mahalanobis distance from the class center
    mahalDist(k, :) = sum(Xnorm.*(C.invRegCov(:,:,k)*Xnorm), 1);
    % Compute the standard Gaussian log-likelihood (based on the
    % regularized covariance matrix)
    out(k, :) = log(C.classProb(k)) - 0.5*C.logDetRegCov(k) - ...
        0.5*mahalDist(k,:);
  else
    out(k,:) = -Inf;
  end
end

if nargout>1,
  out_orig = out;
end

if strcmp(C.rejectMethod, 'threshold'),
  % Rejecting with NaN markers: If *all* classes indicate a very low
  % log-likelihood, we reject. Still, it is unclear whether this should be
  % based on the values in OUT (that include the class probabilities) or on
  % the pure Gaussian likelihoods
  threshold = C.rejectParam;
  reject = all(out<threshold, 1);
  % Mark them with NaNs
  out(:,reject) = NaN;
  % Simplify for 2-class case: out>0 indicates class 2, otherwise class 1
  if nClasses==2,
    out = out(2,:)-out(1,:)+C.bias;
  end
else
  if nClasses~=2,
    error(sprintf('Method ''%s'' can only be used for 2-class problems', C.rejectMethod));
  end
  % singleOut>0 indicates class 2, otherwise class 1
  singleOut = out(2,:)-out(1,:)+C.bias;
  % Index of classifier output for the winning class, for each point:
  winnerInd = (1:2:(2*nPoints))+(singleOut>0);
  switch C.rejectMethod
    case 'classCond'
      % Multiply classifier output with the class-conditional density of the
      % winning class.
      mass = C.rejectParam(1);
      if (mass<=0) | (mass>1),
        error('Probability mass given in rejectParam(1) must be in [0..1]');
      end
      scaling = C.rejectParam(2);
      if (scaling<=0) | (scaling>1),
        error('Scaling constant given in rejectParam(2) must be in [0..1]');
      end
      % Mahalanobis-distance such that the required mass is covered
      mahalThreshold = gaussmass(mass, dim);
      % Scale distance such that exp(-0.5*thresh^2*factor)=scaling
      factor = log(scaling)/(-0.5*mahalThreshold^2);
      out = singleOut.*exp(-0.5*mahalDist(winnerInd)*factor);
    case 'customScaling'
      % Customized transformation function from squared Mahalanobis to
      % some arbitrary scaling:
      scaling = feval(C.rejectParam, mahalDist(winnerInd))
      out = singleOut.*scaling;
    otherwise
      error('Invalid option ''rejectMethod''');
  end
end
