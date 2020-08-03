function [Kderiv,opt,scaling] = kernderiv_rbfratquad(X1,X2,deriv,varargin)

% Author(s): Anton Schwaighofer, Mar 2005
% $Id: kernderiv_rbfratquad.m,v 1.2 2006/06/19 19:59:14 neuro_toolbox Exp $

error(nargchk(3, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'K', [], ...
                        'inweights', ones([1 ndims])*log(1/ndims), ...
                        'degree', 0, ...
                        'rbfweight', 0);

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
    Kderiv = {{'degree'}, {'rbfweight'}, {'inweights', 1}};
  else
    Kderiv = cell([1 ndims+1]);
    Kderiv{1} = {'degree'};
    Kderiv{2} = {'rbfweight'};
    for i = 1:ndims,
      Kderiv{i+2} = {'inweights', i};
    end
  end
  return;
end

% Check whether we should clean the pre-computed data. This should happen
% when starting to compute all the gradients
if opt.resetTransient,
  opt.Krbf = [];
  opt.Krat = [];
  opt.Drat = [];
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

% For all of the derivatives, the actual kernel matrix is
% required. Compute if not passed as an option
if isempty(opt.Krbf) | isempty(opt.Krat),
  Drbf = weightedDist(X1, X2, exp(opt.inweights-log(2)));
  opt.Drat = Drbf*exp(-opt.degree);
  opt.Krbf = exp(-Drbf);
  opt.Krat = (opt.Drat+1).^(-exp(opt.degree));
end

scaling = 1;
switch param
  case 'degree'
    % 'Degree' parameter of the ratquad contribution:
    Kderiv = opt.Krat .* ( log(1 + opt.Drat) - opt.Drat./(1+opt.Drat) );
    scaling = -exp(opt.degree)*sigma(-opt.rbfweight);
  case 'rbfweight'
    % Weight of the RBF part:
    Kderiv = opt.Krbf - opt.Krat;
    scaling = sigma(opt.rbfweight)*sigma(-opt.rbfweight);
  case 'inweights'
    if isempty(opt.F),
      opt.F = opt.Krat./(1 + opt.Drat);
    end
    if scalarInweights,
      % With one shared input weight, we need to compute derivatives in
      % all dimensions and sum up
      Kderiv = weightedDist(X1,X2);
      Kderiv = Kderiv.*((sigma(opt.rbfweight)/2)*opt.Krbf + ...
                        (sigma(-opt.rbfweight)/2)*opt.F);
      scaling = -exp(opt.inweights);
    else
      Kderiv = weightedDist_innerDeriv(X1, X2, dim).*...
               ((sigma(opt.rbfweight)/2)*opt.Krbf + ...
                (sigma(-opt.rbfweight)/2)*opt.F);
      scaling = -exp(opt.inweights(dim));
    end      
  otherwise
    error('Invalid parameter given in argument ''deriv''');
end
if nargout<3,
  % Only one or two return args: Return the full derivative matrix
  Kderiv = scaling*Kderiv;
end


function s = sigma(x)
s = 1./(1+exp(-x));
