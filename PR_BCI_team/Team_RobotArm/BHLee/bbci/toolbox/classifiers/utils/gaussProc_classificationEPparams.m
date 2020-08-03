function [A,mu,invL,evidence] = gaussProc_classificationEPparams(K,nat1Site,nat2Site,nat1Cavity,nat2Cavity,y)
% gaussProc_classificationEPparams - Parameters for EP approximation
%
% Synopsis:
%   [A,mu] = gaussProc_classificationEPparams(K,nat1Site,nat2Site,nat1Cavity,nat2Cavity)
%   [A,mu,invL,evidence] = gaussProc_classificationEPparams(K,nat1Site,nat2Site,nat1Cavity,nat2Cavity,y)
%   
% Arguments:
%  K: [N N] kernel matrix
%  nat1Site: [N 1] vector. Natural parameter 1 of site distributions
%  nat2Site: [N 1] vector. Natural parameter 2 of site distributions
%  nat1Cavity: [N 1] vector. Natural parameter 1 of cavity distributions
%  nat2Cavity: [N 1] vector. Natural parameter 2 of cavity distributions
%  y: [1 N] vector, +1 resp -1 class labels
%   
% Returns:
%  A: [N N] matrix, approximate predictive covariance under EP
%  mu: [N 1] vector, approximate predictive mean under EP
%  invL: [N N] matrix, Cholesky factor of inverse approximate covariance
%      matrix
%  evidence: Scalar. Log marginal likelihood with EP approximation
%   
% Description:
%   Helper subroutine for gaussProc_classificationEP
%   
%
% See also: gaussProc_classificationEP
% 

% Author(s), Copyright: Anton Schwaighofer, Jun 2006
% $Id: gaussProc_classificationEPparams.m,v 1.1 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(3,6, nargin));
if nargout>3 & nargin<6,
  error('To compute evidence, label vector y must be given as input');
end
m = size(K,1);
v = sqrt(nat1Site(:));
L = chol(eye(m) + (v*v').*K)';
invL = chol2invChol(L);
U = (K*spdiags(v,0,m,m) * invL');
A = K - U*U';
mu = A*nat2Site;
if nargout>3,
  varCavity = 1./nat1Cavity;
  muCavity = nat2Cavity.*varCavity;
  tmp1 = 0.5 * sum(log(1+nat1Site./nat1Cavity)) - sum(log(diag(L)));
  tmp2 = -0.5 * nat2Site' * (nat2Site./(nat1Site+nat1Cavity) - mu);
  tmp3 = 0.5 * sum(muCavity.*nat1Cavity.*(nat1Site.*muCavity-2*nat2Site)./(nat1Site+nat1Cavity));
  tmp4 = sum(log(erfc(-(y'.*muCavity)./sqrt(1+varCavity)/sqrt(2))/2));
  evidence = -(tmp1 + tmp2 + tmp3 + tmp4);
end
