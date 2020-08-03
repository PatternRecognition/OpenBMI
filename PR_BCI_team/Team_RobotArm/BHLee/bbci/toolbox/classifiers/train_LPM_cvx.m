function LPM= train_LPM_cvx(data, labels, varargin)
% train_LPM - train a linear programming machine. Use
% 'apply_separatingHyperplane' for testing.
%
% Synopsis:
%   LPM = train_LPM(data, labels, properties)
%   LPM = train_LPM(data, labels, 'Property', Value, ...)
%
% Arguments:
%   data:	[n,d]	real valued training data containing n examples of d dims
%   labels:	[1,n]	vector with +1/-1 entries. First class is assumed
%                       to have labels -1, the second class: +1. 
% 
% Returns:
%   LPM:        a trained LPM structure with the fields:
%               .w       - hyperplane vector
%               .b       - threshold
%
% Properties:
%	'C': scalar. Regularization parameter C
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
% 2008 haufe, grozea

% Backward compatibility for the old signature
if nargin == 3 && ~isstruct(varargin{1}) && ~ischar(varargin{1})
  opt = propertylist2struct('C',varargin{1},'C_weight',1);
elseif nargin == 4 && ~ischar(varargin{1})
  opt = propertylist2struct('C',varargin{1},'C_weight',varargin{2});
else % new signature
  defaults = {'C',1,'C_weight',1};
  opt = set_properties(varargin, defaults);
end

[K,N]= size(data);

X=data(:,find(labels(1,:)>=0.5));N=size(X,2);
Y=data(:,find(labels(1,:)<0.5));M=size(Y,2);
%g=1e-3;
cvx_begin
    cvx_quiet(true)
    variables w(K) b(1) u(N) v(M);
    minimize (norm(w, 1) + opt.C*(sum(u) + sum(v)))
    X'*w - b >= 1 - u;
    Y'*w - b <= -(1 - v);
    u >= 0;
    v >= 0;
cvx_end;

LPM.w = - w;
LPM.b = b; 

