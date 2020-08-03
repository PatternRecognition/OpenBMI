function FD= train_FDqwqx(xTr, yTr, varargin)
%FD= train_FDqwlq(xTr, yTr, C)
%
% mathematical programming of Fisher discriminant,
% quadratic w - quadratic xi minimization (SMika):
%
%   min. ||w||^2 + C*mean xi_k^2   s.t. ....
%
%   w'*xTr(:,k) + b =  1-xi(k)  for class 1 (yTr(k)==1)
%                   = -1+xi(k)  for class 2 (yTr(k)==-1)
%
% C can be an element between 0 and infinity
%
% GLOBZ  LPENV

% Backward compatibility for the old signature
if nargin == 3 && ~isstruct(varargin{1}) && ~ischar(varargin{1})
  opt = propertylist2struct('C',varargin{1});
else % new signature
  defaults = {'C',1};
  opt = set_properties(varargin, defaults);
end

start_cplex;

if size(yTr,1)==2, yTr= [-1 1]*yTr; end
[N,K]= size(xTr);

% [w, xi, b]
Q  = sparse([speye(N, N+K+1); ...
             spalloc(K, N, 0), opt.C/K*speye(K, K+1); ...
             spalloc(1, N+K+1, 0)]);
c  = zeros(N+K+1,1);
lb = -inf*ones(N+K+1,1);
ub = inf*ones(N+K+1,1);

A  = sparse([spdiag(yTr)*xTr', speye(K), yTr']);
a= (yTr==1)/sum(yTr==1)+(yTr==-1)/sum(yTr==-1);
a=a'*K;
neq= K;

[opt, lambda, how]= qp_solve(LPENV, Q, c, A, a, lb, ub, neq, 0);
if ~isequal(how,'OK'),
  fprintf(2, 'NOT OKI %s', how);
end
  
FD.w= opt(1:N);
FD.b= -mean(xTr,2)'*FD.w;   
%threshold might be inappropriate
