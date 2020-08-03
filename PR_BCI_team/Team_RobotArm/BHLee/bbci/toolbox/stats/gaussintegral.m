function v = gaussintegral(t,p)
% gaussintegral - Definite integral of multivariate Gaussian density function
%
% Synopsis:
%   v = gaussintegral(t,p)
%   
% Arguments:
%   t: Integration bounds. Integration is within a circle of radius r from the
%      mean. t can be a vector, containing several integration bounds.
%   p: Dimensionality of the Gaussian pdf
%   
% Returns:
%   v: Value of the integral, that is, probability mass that is contained in a
%      circle of radius t from the mean.
% 
% Description:
%   This function computes the definite integral
%                 /
%   (2*pi)^(-p/2) | exp(-1/2 norm(x)^2) dx
%                 /R
%   where the integration region R is a circle of radius t, that is, the
%   region norm(x)<=t. x is a vector of dimensionality p.
%
% Examples:
%   v = gaussintegral(2,1)
%   returns the probability mass contained in a 2-dimensional Gaussian
%   density within a circular region of radius 1.
%   
% See also: gammainc,erf,gaussmass
% 

% Author(s): Anton Schwaighofer, Oct 2004
% $Id: gaussintegral.m,v 1.1 2005/06/30 17:20:47 neuro_toolbox Exp $

error(nargchk(2, 2, nargin));

if prod(size(p))~=1 | p<0,
  error('Dimensionality p must be a positive scalar');
end
% Assume zero for every negative input argument. Could leave that
% unrestricted, but this way it is nicer (and may cause fewer problems
% with gaussmass.m, when searching for zeros)
v = zeros(size(t));
positiveT = t>0;
t = t(positiveT);

% Empirically verified that the equations also hold for p==1

% The god-given (Mathematica-given) constant
expConstant = 0.34657359027997264311;
logScaling = expConstant*p - p/2*log(2);
v(positiveT) = exp(logScaling)*gammainc((t.^2)/2, p/2);
