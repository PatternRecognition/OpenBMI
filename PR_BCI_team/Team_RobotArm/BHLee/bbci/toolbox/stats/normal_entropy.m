function H = normal_entropy(cov1,mean1,cov2,mean2)
% normal_entropy - Entropy and relative entropy of (multivariate) normal distribution
%
% Synopsis:
%   H = normal_entropy(cov1)
%   D = normal_entropy(cov1,mean1,cov2,mean2)
%   
% Arguments:
%  cov1: [dim dim] matrix. Covariance matrix of distribution 1
%  mean1: [dim 1] vector. Mean of distribution 1
%  cov2: [dim dim] matrix. Covariance matrix of distribution 2
%  mean2: [dim 1] vector. Mean of distribution 2
%   
% Returns:
%  H: Scalar. Entropy of distribution 1 (if 1 argument provided)
%  D: Scalar. Relative entropy between distributions 1 and 2 (if 4 arguments provided)
%   
% Description:
%   If only one argument is provided, the routine computes the differential
%   entropy of distribution 1, defined as
%     H = 0.5 * log(det(2*pi*exp(1)*cov1))
%   If 4 arguments are provided, the relative entropy is computed,
%     D(N(mean1,cov1) || N(mean2,cov2)) = 
%     = 0.5*log(inv(cov1)*cov2) + 0.5*trace(inv(inv(cov1)*cov2)-eye)
%       + 0.5*(mean1-mean2)*inv(cov2)*(mean1-mean2)
%   The relative entropy is equivalent to the Kullback-Leibler distance
%   between the two probability distributions.
%
%   Mind that lots of inverses are computed here. For poorly conditioned
%   covariance matrices, this can lead to problems, as no particular
%   measures to ensure numerical stability are taken.
%
%   In contrast to most other routines for the normal distribution, this
%   routine assumes matrix inputs to correspond to the multivariate case,
%   whereas routines such as normal_cdf or normal_pdf assume univariate
%   distributions also for matrix inputs!
%
% Examples:
%   The entropy of the standard normal distribution with mean 0, variance
%   one is
%     normal_entropy(1)
%     ans = 1.4189
%   The entropy of a bivariate Gaussian, where the two components are
%   independent, is simply double the contributions of the two
%   components:
%     normal_entropy(eye(2))
%     ans = 2.8379
%   Relative entropy between unit variance Gaussian centered at -1 and 0:
%     normal_entropy(1, -1, 1, 0)
%       ans = 0.5
%   Mind that the relative entropy is not symmetric:
%     normal_entropy(1, 0, 2, -1)
%       ans = 0.34657
%     normal_entropy(2, 0, 1, -1)
%       ans = 0.65343
%
%   
% See also: normal_pdf
% 

% Author(s), Copyright: Anton Schwaighofer, May 2006
% $Id: normal_entropy.m,v 1.2 2006/06/19 20:26:11 neuro_toolbox Exp $

error(nargchk(1, 4, nargin));
[dim1,dim2] = size(cov1);
if dim1~=dim2,
  error('Input argument cov1 must be a square matrix');
end
if nargin==1,
  H = 0.5*(dim1*log(2*pi*exp(1)) + logdet(cov1));
elseif nargin==4,
  [dim1,dim2] = size(cov2);
  if dim1~=dim2,
    error('Input argument cov2 must be a square matrix');
  end
  if length(mean1)~=dim1,
    error('Input argument mean1 must be a [dim 1] vector');
  end
  if length(mean2)~=dim1,
    error('Input argument mean2 must be a [dim 1] vector');
  end
  mean1 = mean1(:);
  mean2 = mean2(:);
  C = inv(cov1)*cov2;
  H = 0.5*(logdet(C) + (trace(inv(C))-dim1) + (mean1-mean2)'*inv(cov2)*(mean1-mean2));
else
  error('normal_entropy can only be used with either 1 or 4 input arguments');
end


function D = logdet(A)
L = chol(A);
D = 2*sum(log(diag(L)));
