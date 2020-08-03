function FD= train_FDqwlx(xTr, yTr, C)
%FD= train_FDqwlx(xTr, yTr, C)
%
% mathematical programming of Fisher discriminant,
% quadratic w - linear xi minimization (SMika)
%
%   min. ||w||^2 + C*mean |xi_k|   s.t. 
%
%   w'*xTr(:,k) + b =  1-xi(k)  for class 1 (yTr(k)==1)
%                   = -1+xi(k)  for class 2 (yTr(k)==-1)
%
% C can be an element between 0 and infinity
%
% GLOBZ  LPENV

start_cplex

if size(yTr,1)==2, yTr= [-1 1]*yTr; end
[N,K]= size(xTr);

% [w, xi+, xi-, b]
Q  = sparse([speye(N, N+2*K+1); spalloc(2*K+1, N+2*K+1, 0)]);
c  = [zeros(N,1); C/K*ones(2*K,1); 0];
lb = [-inf*ones(N,1); zeros(2*K,1); -inf];
ub = inf*ones(N+2*K+1, 1);

A  = sparse([spdiag(yTr)*xTr', speye(K), -speye(K), yTr']);
a= (yTr==1)/sum(yTr==1)+(yTr==-1)/sum(yTr==-1);
a=a'*K;
neq= K;

[opt, lambda, how]= qp_solve(LPENV, Q, c, A, a, lb, ub, neq, 0);
if ~isequal(how,'OK'),
  fprintf(2, 'NOT OKI %s', how);
end
  
FD.w= opt(1:N);
FD.b= opt(end);   
%threshold might be inappropriate
