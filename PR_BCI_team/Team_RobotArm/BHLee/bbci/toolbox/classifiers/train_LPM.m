function LPM= train_LPM(data, labels, varargin)
% train_LPM - train a linear programming machine. Use
% 'apply_separatingHyperplane' for testing.
%
% Synopsis:
%   LPM = train_LPM(data, labels, properties)
%   LPM = train_LPM(data, labels, 'Property', Value, ...)
%
% Arguments:
%   data:	[d,n]	real valued training data containing n examples of d dims
%   labels:	[2,n]	logical labels (2-class problems only!) or [1 d]
%                       vector with +1/-1 entries. First class is assumed
%                       to have labels -1, the second class: +1. 
% 
% Returns:
%   LPM:        a trained LPM structure with the fields:
%               .w       - hyperplane vector
%               .b       - threshold
%               .lambda  - Lagrange multiplier vector
%
% Properties:
%	'C': scalar or [1 2] matrix. Regularization parameter C
%                       If C is a [1 2] matrix, class wise regularization
%                       will be used, with C(1) the parameter for class 1
%                       (the "-1 class") and C(2) for class 2 (the "+1 class")
%
%       'C_weight'      weight by which the regularization parameter C will 
%                       be multiplied for class 1 (label == -1). Only
%                       used when length(C) == 1. 
%
% Description:
%
% Training of an LPM is expressed by the following optimization problem:
%
% minimizing |w|_1m + C*|xi|_1m,  where |x|_1m:= mean(abs(x)) and
%
% w'*data(:,k) + b >=  1-xi(k)  for class 2 (labels(k)==1)
%                  <= -1+xi(k)  for class 1 (labels(k)==-1) and
%            xi(k) >= 0         for all k
%
% If C_weight is given, the norm of the slacks for each classes is
% penalized individually by the respective value as follows:
%    C        for yTr==1, 
%    C*weight for yTr==-1
% If C_weight is not given the "errors" from each class are treated equally.
%
% GLOBZ LPENV

% Backward compatibility for the old signature
if nargin == 3 && ~isstruct(varargin{1}) && ~ischar(varargin{1})
  opt = propertylist2struct('C',varargin{1},'C_weight',1);
elseif nargin == 4 && ~ischar(varargin{1})
  opt = propertylist2struct('C',varargin{1},'C_weight',varargin{2});
else % new signature
  defaults = {'C',1,'C_weight',1};
  opt = set_properties(varargin, defaults);
end


start_cplex;

if size(labels,1)==2, labels= [-1 1]*labels; end
INF= inf;
[N,K]= size(data);

% [w+, w-, xi, b]
cv = zeros(K,1);
if size(opt.C) == 2
  cv(find(labels == -1)) = opt.C(1)/K;
  cv(find(labels == 1)) = opt.C(2)/K;
else
  cv = opt.C/K * ones(K,1);
  ind = find(labels == -1);
  cv(ind) = opt.C_weight*cv(ind);
end

c  = [1/N*ones(2*N,1); cv; 0];

lb = [zeros(2*N+K,1); -INF];
ub = INF*ones(2*N+K+1, 1);

A  = -sparse([spdiag(labels)*data', -spdiag(labels)*data', speye(K), labels']);
a  = -ones(K,1);
neq= 0;

[opt, lambda, how]= lp_solve(LPENV, c, A, a, lb, ub, neq, 0);
if ~isequal(how,'OK'),
  fprintf(2, 'NOT OKI %s', how);
end
  
LPM.w = opt(1:N) - opt(N+1:2*N);
LPM.b = opt(end);
%LPM.xi = opt(N+1:N+K) - opt(N+K+1:N+2*K);
LPM.lambda = lambda;
