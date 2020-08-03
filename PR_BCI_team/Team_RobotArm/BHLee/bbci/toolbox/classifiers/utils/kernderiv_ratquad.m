function [Kderiv,opt,scaling] = kernderiv_ratquad(X1,X2,deriv,varargin)
% kernderiv_ratquad - Kernel derivative for the rational quadratic kernel
%
% Synopsis:
%   Kderiv = kernderiv_ratquad(X1,X2,deriv)
%   [Kderiv,opt] = kernderiv_ratquad(X1,X2,deriv,Property,Value,...)
%   
% Arguments:
%   X1: [d N1] matrix. Input data 1 for kernel matrix computation
%   X2: [d N2] matrix. Input data 2 for kernel matrix computation
%   deriv: String or cell array. This chooses the kernel parameter for which
%       to compute the derivative. With deriv=='returnParams', the output is
%       not a kernel derivative matrix, but rather a cell array of cell
%       arrays containing all possible values for 'deriv'.
%   
% Returns:
%   Kderiv: [N1 N2] matrix with the derivative of the kernel matrix with
%       respect to the chosen parameter.
%   opt: Struct array, with Property/value pairs converted to structure fields.
%   
% Description:
%   
%   
%   
% Examples:
%   
%   
% See also: kern_ratquad
% 

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: kernderiv_ratquad.m,v 1.2 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(3, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'K', [], ...
                        'inweights', ones([1 ndims])*log(1/ndims), ...
                        'degree', 0);

if ~isempty(X2),
  if ndims~=size(X2,1),
    error('Number of dimensions in X1 and X2 must match');
  end
  N2 = size(X2,2);
else
  N2 = N1;
end
scalarInweights = prod(size(opt.inweights))==1;
if ischar(deriv) & strcmp(lower(deriv), 'returnparams'),
  % Generate cell array with all allowed parameters
  if scalarInweights,
    Kderiv = {{'degree'}, {'inweights', 1}};
  else
    Kderiv = cell([1 ndims+1]);
    Kderiv{1} = {'degree'};
    for i = 1:ndims,
      Kderiv{i+1} = {'inweights', i};
    end
  end
  return;
end

% Check whether we should clean the pre-computed data. This should happen
% when starting to compute all the gradients
if opt.resetTransient,
  opt.K = [];
  opt.D = [];
  opt.F = [];
  opt.resetTransient = 0;
end

if iscell(deriv),
  param = deriv{1};
  if length(deriv)>1,
    dim = deriv{2};
  else
    dim = [];
  end
else
  param = deriv;
  dim = [];
end

% For most of the derivatives, the actual kernel matrix is
% required. Compute if not passed as an option
if isempty(opt.K),
  [opt.K, opt.D] = kern_ratquad(X1, X2, opt);
end

switch param
  case 'degree'
    % Derivative for the parametrization weight/degree:
    Kderiv = opt.K .* ( log(1 + opt.D) - opt.D./(1+opt.D) );
    % Derivative for the original parametrization (1+z)^(-degree):
% $$$     Kderiv = opt.K .*log(1 + opt.D);
    scaling = -exp(opt.degree);
  case 'inweights'
    if isempty(opt.F),
      opt.F = opt.K./(1 + opt.D);
    end
    if scalarInweights,
      % With one shared input weight, we need to compute derivatives in
      % all dimensions and sum up
      Kderiv = weightedDist(X1,X2);
      Kderiv = Kderiv.*opt.F;
      scaling = -exp(opt.inweights-log(2));
    else
      Kderiv = opt.F.*weightedDist_innerDeriv(X1, X2, dim);
      scaling = -exp(opt.inweights(dim)-log(2));
    end      
  otherwise
    error('Invalid parameter given in argument ''deriv''');
end
if nargout<3,
  % Only one or two return args: Return the full derivative matrix
  Kderiv = scaling*Kderiv;
end
