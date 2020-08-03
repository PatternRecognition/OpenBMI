function t = gaussmass(v, p, varargin)
% gaussmass - Find a region of given probability mass for multivariate Gaussian
%
% Synopsis:
%   t = gaussmass(v,p)
%   t = gaussmass(v,p,'Parameter',value,...)
%   
% Arguments:
%   v: Probability mass (value between 0 and 1)
%   p: Dimensionality of the Gaussian pdf
%   All additional arguments are passed on directly to function
%   optimset.
%   
% Returns:
%   t: Radius of a circular region around the mean that contains a probability
%       mass of v.
%   
% Description:
%   This function uses fzero to find the point where function
%   gaussintegral.m returns v. fzero is called with its default
%   options. Additional arguments passed to gaussmass are passed on to
%   optimset, the resulting options structure is passed on to fzero.
%
% Examples:
%   t = gaussmass(0.95, 2)
%   returns the radius of the disk that contains 95% of the mass of a
%   bivariate Gaussian distribution.
%   
%   
% See also: gaussintegral,erfinv,fzero,optimset
% 

% Author(s): Anton Schwaighofer, Oct 2004
% $Id: gaussmass.m,v 1.1 2005/06/30 17:20:47 neuro_toolbox Exp $

if prod(size(v))~=1 | v<0 | v>1,
  error('Probability mass v must be a scalar in the range [0..1]');
end
if prod(size(p))~=1 | p<0,
  error('Dimensionality p must be a positive scalar');
end

% Also handle the case p==1 with fzero. There should be a way with
% erfinv, but the scaling is somehow wrong
% $$$ if p==1,
% $$$   t = erfinv(v);
% $$$ else

% Choose starting point at the maximum of the integrand r^(p-1)exp(-0.5r^2)
start = sqrt(p-1);
options = optimset([], varargin{:});
% Start the search, pass parameters p and v directly on to gausshelper
t = fzero(@gausshelper, start, options, p, v);


function f = gausshelper(t, p, v)

f = gaussintegral(t,p) - v;
