function [Kderiv,opt,scaling] = kernderiv_neuralnet(X1,X2,deriv,varargin)
% kernderiv_neuralnet - Kernel derivative for Radial Basis Function RBF kernel
%
% Synopsis:
%   Kderiv = kernderiv_neuralnet(X1,X2,deriv)
%   [Kderiv,opt] = kernderiv_neuralnet(X1,X2,deriv,Property,Value,...)
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
% See also: kern_neuralnet
% 

% Author(s): Joaquin Quiñonero Candela, Mar 2006
% $Id: kernderiv_neuralnet.m,v 1.1 2006/06/19 19:59:14 neuro_toolbox Exp $


error(nargchk(3, inf, nargin));
[ndims N1] = size(X1);
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'K', [], 'inweights', ones([1 ndims])*log(1/ndims), ...
                        'bias', log(1));

if ~isempty(X2),
  if ndims~=size(X2,1),
    error('Number of dimensions in X1 and X2 must match');
  end
  N2 = size(X2,2);
else
  N2 = N1;
  X2 = X1;
end
scalarInweights = prod(size(opt.inweights))==1;
if ischar(deriv) & strcmp(lower(deriv), 'returnparams'),
  % Generate cell array with all allowed parameters
  if scalarInweights,
    Kderiv = {{'inweights', 1}};
  else
    Kderiv = cell([1 ndims]);
    for i = 1:ndims,
      Kderiv{i} = {'inweights', i};
    end
  end
  Kderiv{end+1} = {'bias', 1};
  scaling = [];
  return;
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

% We need this matrix in any case for the derivatives
D = weightedInnerProd(X1,X2,exp(opt.inweights));


% Check whether we should clean the pre-computed data. This should happen
% when starting to compute all the gradients
if opt.resetTransient,
  opt.K = [];
  opt.D = [];
  opt.Z = [];
  opt.resetTransient = 0;
end

% For most of the derivatives, the actual kernel matrix is
% required. Compute if not passed as an option
if isempty(opt.K),
  opt.K = (exp(opt.bias)+D)./(sqrt(1+exp(opt.bias)+diag(D))*sqrt(1+ ...
                                                    exp(opt.bias)+diag(D)'));
  opt.K = asin(opt.K);
end

if isempty(opt.Z)
  opt.Z = (exp(opt.bias)+D)./(sqrt(1+exp(opt.bias)+diag(D))*sqrt(1+ ...
                                                    exp(opt.bias)+diag(D)'));
end

if isempty(opt.D)
  opt.D = weightedInnerProd(X1,X2,exp(opt.inweights));
end

if isempty(X2)
  d1 = diag(opt.D);
  d2 = diag(opt.D);
else
  d1 = weightedDist(X1,zeros(size(X1)),exp(opt.inweights));
  d2 = weightedDist(X2,zeros(size(X1)),exp(opt.inweights));
end
switch param
 case 'inweights'
  if scalarInweights,
    % With one shared input weight, we need to compute derivatives in
    % all dimensions and sum up
    v1 = sum(X1.^2,1)./(1+exp(opt.bias)+diag(opt.D)');
    v2 = sum(X2.^2,1)./(1+exp(opt.bias)+diag(opt.D)');
    Kderiv = (X1'*X2./(sqrt(1+exp(opt.bias)+diag(opt.D))*sqrt(1+ ...
                                  exp(opt.bias)+ diag(opt.D)'))- ...
              opt.Z.*(repmat(v1',1,N1)+repmat(v2,N1,1))/2)./sqrt(1-opt.Z.^2); 
    scaling = exp(opt.inweights(dim));
  else
    v1 = X1(dim,:).^2./(1+exp(opt.bias)+diag(d1)');
    v2 = X2(dim,:).^2./(1+exp(opt.bias)+diag(d2)');
    Kderiv = (X1(dim,:)'*X2(dim,:)./(sqrt(1+exp(opt.bias)+diag(d1))* ...
                                     sqrt(1+exp(opt.bias)+diag(d2)'))- ...
              opt.Z.*(repmat(v1',1,N1)+repmat(v2,N1,1))/2)./sqrt(1-opt.Z.^2); 
    scaling = exp(opt.inweights(dim));
  end      
 case 'bias'
  v = 1./(1+exp(opt.bias)+diag(opt.D)');
  Kderiv = (1./(sqrt(1+exp(opt.bias)+diag(opt.D))*sqrt(1+exp(opt.bias)+ ...
                                                    diag(opt.D)'))- ...
            opt.Z.*(repmat(v',1,N1)+repmat(v,N1,1))/2)./sqrt(1-opt.Z.^2); 
  scaling = exp(opt.bias);
 otherwise
  error('Invalid parameter given in argument ''deriv''');
end
if nargout<3,
  % Only one or two return args: Return the full derivative matrix
  Kderiv = scaling*Kderiv;
end

return

