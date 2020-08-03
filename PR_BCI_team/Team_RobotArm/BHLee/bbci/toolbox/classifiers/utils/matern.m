function [M, dnu, dZ] = matern(nu, Z, sigma, M)
% matern - Matern covariance function and its derivatives
%
% Synopsis:
%   M = matern(nu,Z)
%   [M,dnu,dZ] = matern(nu,Z,sigma,M)
%   
% Arguments:
%   nu: Positive scalar. Degree of the Matern function
%   Z: [N1 N2] matrix. Distance argument of the Matern function. The Matern
%       function is computed for each entry of Z.
%   sigma: Positive scalar. Scaling parameter. The Matern function is scaled
%       such that matern(nu,0,sigma) gives sigma. Default value: 1
%   M: [N1 N2] matrix. Computing Matern derivatives requires computing
%       matern(nu,Z,sigma). If this is already available from a previous call, it
%       can be passed as the argument M.
%   
% Returns:
%   M: [N1 N2] matrix. Result of evaluating the Matern function for each entry
%       of Z.
%   dnu: [N1 N2] matrix. Derivative of the Matern function with respect to the
%       degree nu, evaluated at each entry of Z
%   dZ: [N1 N2] matrix. Derivative of the Matern function with respect to the
%       input argument Z, for each entry of Z.
%   
% Description:
%   The Matern class of functions is a covariance function where the
%   degree of smoothness can be varied continuously by its parameter
%   nu. nu=0 gives infinitely rough sample paths, the Matern function
%   corresponds to the Ornstein-Uhlenbeck process in this case. 
%   nu->inf gives sample paths that are infinitely times differentiable
%   (infinitely smooth). For nu->inf, the Matern function corresponds to
%   the radial basis function ("Gaussian") kernel.
%   
%   
% Examples:
%   When computing derivatives, save some computation time by
%   pre-computing the Matern function:
%     m = matern(nu, z, sigma);
%     [m, dnu, dz] = matern(nu, z, sigma, m);
%   Plot the shape of the Matern function for different nu:
%     z = 0:0.05:2;
%     m1 = matern(2, z); m2 = matern(5, z); m3 = matern(10, z);
%     plot(z, m1, 'k-', z, m2, 'b-', z, m3, 'r-');
%     legend('nu = 2', 'nu = 5', 'nu = 10');
%   
% References:
%   Stein, Michael: Interpolation of Spatial Data. Springer, 1999
%   
% See also: kern_matern,gammaln,besselk,psi
% 

% Author(s), Copyright: Anton Schwaighofer, Mar 2003
% $Id: matern.m,v 1.1 2006/06/19 20:17:23 neuro_toolbox Exp $

error(nargchk(2, 4, nargin));
if nargin<4,
  M = [];
end
if nargin<3,
  sigma = [];
end
if isempty(sigma),
  sigma = 1;
end
if prod(size(nu))~=1,
  error('Smoothness paramter NU must be a scalar');
end

w = warning;
warning('off');
Zscaled = (2*sqrt(nu)).*Z;
logZscaled = log(Zscaled);
warning(w);

% Re-compute Matern function only if not provided as input or if M is the
% only output argument
if isempty(M) | nargout<2,
  % Standard version
  % p1 = sigma/(2^(nu-1)*gamma(nu));
  % p2 = Zscaled.^nu;
  % p3 = besselk(nu,Zscaled);
  % M = p1.*p2.*p3;
  % M(isnan(M))=sigma;
  
  % This is supposed to be more stable for large values of nu
  logM1 = log(sigma)-(nu-1)*log(2)-gammaln(nu);
  % Compute Bessel functions scaled by EXP(ZSCALED). Need to subtract
  % ZSCALED on the log scale later
  warning('off');
  logM2 = log(besselk(nu, Zscaled, 1));
  warning(w);
  % For large degree NU, BESSELK will give Inf. Z==0 may cancel this,
  % if not: issue a warning. Anyway we will pretend that Infs are
  % cancelled by setting any of them to SIGMA
  besselInf = find(isinf(logM2));
  if any(Zscaled(besselInf)>0),
    warning(sprintf(['Some of the Bessel functions are Infinite. Degree\n'...
                     'of Matern function NU is probably too large']));
  end
  M = exp(logM1+logM2+nu*logZscaled-Zscaled);
  M(isnan(M)) = sigma;
  M(isinf(M)) = sigma;
end

if nargout>=2,
  % First term of the derivate of the Matern function with respect to
  % degree NU. LOG(SQRT(NU)*Z) replaced by LOG(2*SQRT(NU)*Z)-LOG(2)
  dnu1 = M.*((0.5-log(2)-psi(nu))+logZscaled);
  % Second term, sum of the Bessel and Bessel derivative functions. Compute
  % with scaling EXP(ZSCALED), all Bessel evaluations have the same argument
  besselPM = besselk(nu+1, Zscaled, 1) + besselk(nu-1, Zscaled, 1);
  besselSum = -Z.*besselPM./(2*sqrt(nu)) + dbesselk(nu, Zscaled, 1);
  warning('off');
  dnu2 = sign(besselSum).*exp((log(sigma)+log(2)-nu.*log(2)-gammaln(nu))+ ...
                              nu.*logZscaled+log(abs(besselSum))-Zscaled);
  warning(w);
  dnu = dnu1+dnu2;
  dnu(isinf(dnu)) = 0;
  dnu(isnan(dnu)) = 0;
end

if nargout>=3,
  % First term of the derivate of the Matern function with respect to its
  % argument Z. This is simply a scaled version of the already computed
  % evaluation of Matern on all distances
  warning('off');
  dZ1 = nu.*(M./Z);
  % Second term of the derivative. Use the already computed Bessel functions
  % of degrees NU+1 and NU-1, don't forget to subtract the scaling
  % term. Again, replace LOG(SQRT(NU)*Z) by LOG(2*SQRT(NU)*Z)-LOG(2)
  dZ2 = exp((log(sigma)+log(2)+0.5*log(nu)-nu.*log(2)-gammaln(nu)) + ...
            nu.*logZscaled + log(besselPM)-Zscaled);
  warning(w);
  dZ = dZ1 - dZ2;
  dZ(isinf(dZ)) = 0;
  dZ(isnan(dZ)) = 0;
end


function d = dbesselk(nu, z, scale, dnu)
% DBESSELK - Numerical gradient of BESSELK with respect to degree NU
%
%   D = DBESSELK(NU, Z)
%   Compute the derivate of the modified Bessel function (second kind)
%   with respect to the degree NU, evaluated at Z. Since there is no easy
%   closed form expression for this gradient, this is computed
%   numerically.
%   DBESSELK(NU, Z, 1) scales the gradient by EXP(Z).
%   DBESSELK(NU, Z, SCALE, DNU) uses a step size of DNU to find the
%   numerical gradient. Default value: NU*1E-6
%   
%   See also BESSELK
%

% Copyright (c) by Anton Schwaighofer (2003)

error(nargchk(2, 4, nargin));
if nargin<4,
  dnu = nu*1e-6;
end
if nargin<3,
  scale = 0;
end
d = (besselk(nu+dnu,z,scale)-besselk(nu-dnu,z,scale))./(2*dnu);
