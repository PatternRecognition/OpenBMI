function [Kderiv,opt,scaling] = kernderiv_const(X1,X2,deriv,varargin)

error(nargchk(3, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});

if ~isempty(X2),
  if ndims~=size(X2,1),
    error('Number of dimensions in X1 and X2 must match');
  end
  N2 = size(X2,2);
else
  N2 = N1;
end

if ischar(deriv) & strcmp(lower(deriv), 'returnparams'),
  % We don't have any params....
  Kderiv = {};
  return;
end

scaling = 0;
Kderiv = zeros([N1 N2]);

if nargout<3,
  % Only one or two return args: Return the full derivative matrix
  Kderiv = scaling*Kderiv;
end
