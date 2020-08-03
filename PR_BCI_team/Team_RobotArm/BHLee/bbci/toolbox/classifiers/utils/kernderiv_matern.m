function [Kderiv,opt,scaling] = kernderiv_matern(X1,X2,deriv,varargin)


error(nargchk(3, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'K', [], ...
                        'derivZZ', [], ...
                        'inweights', zeros([1 ndims]), ...
                        'smoothness', 1);

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
  Kderiv = {};
  % We should ensure that smoothness derivatives are computed first: Some
  % of the results computed there will be needed later for input weight
  % derivatives
  Kderiv{1} = {'smoothness'};
  % Generate cell array with all allowed parameters
  if scalarInweights,
    Kderiv{end+1} = {'inweights', 1};
  else
    for i = 1:ndims,
      Kderiv{end+1} = {'inweights', i};
    end
  end
  return;
end

% Check whether we should clean the pre-computed data. This should happen
% when starting to compute all the gradients
if opt.resetTransient,
  opt.derivZZ = [];
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
  opt.K = kern_matern(X1, X2, opt);
end
scaling = 1;
switch param
  case 'smoothness'
    [dnu, dzz] = maternderivs(X1, X2, opt);
    % Derivative with respect to degree nu = exp(opt.smoothness): 
    Kderiv = dnu;
    scaling = exp(opt.smoothness);
    % Pass this out to the caller, so that we can re-use it later
    opt.derivZZ = dzz;
  case 'inweights'
    if isempty(opt.derivZZ),
      [dnu, dzz] = maternderivs(X1, X2, opt);
      opt.derivZZ = dzz;
    end
    if scalarInweights,
      % With one shared input weight, we need to compute derivatives in
      % all dimensions and sum up
      Kderiv = 0;
      for i = 1:ndims,
        Kderiv = Kderiv+weightedDist_innerDeriv(X1, X2, i);
      end
      Kderiv = -Kderiv.*opt.derivZZ;
      scaling = -0.5*exp(opt.inweights);
    else
      Kderiv = -opt.derivZZ.*weightedDist_innerDeriv(X1, X2, dim);
      scaling = -0.5*exp(opt.inweights(dim));
    end      
  otherwise
    error('Invalid parameter given in argument ''deriv''');
end
if nargout<3,
  % Only one or two return args: Return the full derivative matrix
  Kderiv = scaling*Kderiv;
end


function [dnu, dzz] = maternderivs(X1, X2, opt)
% Need to re-compute the weighted distances z
z = sqrt(weightedDist(X1, X2, exp(opt.inweights)));
% Compute the derivatives of the Matern function with respect to
% degree NU and argument Z. DZ will be needed when computing
% derivatives with respect to input weights
[dummy, dnu, dz] = matern(exp(opt.smoothness), z, 1, opt.K);
warnstate = warning;
warning off;
dzz = dz./z;
% Derivatives at z=0 should also be zero, thus we set the 0/0=NaN
% results to 0
dzz(isnan(dzz)) = 0;
warning(warnstate);
