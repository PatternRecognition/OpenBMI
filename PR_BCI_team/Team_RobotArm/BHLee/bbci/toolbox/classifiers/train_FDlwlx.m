function FD= train_FDlwlx(xTr, yTr, C)
%FD= train_FDlwlx(xTr, yTr, C)
%
% mathematical programming of Fisher discriminant,
% linear w - linear xi minimization (SMika)
%
%  min. ||w||_1 + C*mean |xi_k|   s.t.
%
%   w'*xTr(:,k) + b =  1-xi(k)  for class 1 (yTr(k)==1)
%                   = -1+xi(k)  for class 2 (yTr(k)==-1)
%
% C can be an element between 0 and infinity
%
% GLOBZ  LPENV

start_cplex;

if size(yTr,1)==2, yTr= [-1 1]*yTr; end
INF= inf;
[N,K]= size(xTr);

% [w+, w-, xi+, xi-, b]
c  = [1/N*ones(2*N,1); C/K*ones(2*K,1); 0];
lb = [zeros(2*N+2*K,1); -INF];
ub = INF*ones(2*N+2*K+1, 1);

A  = sparse([spdiag(yTr)*xTr', -spdiag(yTr)*xTr', speye(K), -speye(K), yTr']);
a= (yTr==1)/sum(yTr==1)+(yTr==-1)/sum(yTr==-1);
a=a'*K;
neq= K;

[opt, lambda, how]= lp_solve(LPENV, c, A, a, lb, ub, neq, 0);
if ~isequal(how,'OK'),
  fprintf(2, 'NOT OKI %s', how);
end
  
FD.w= opt(1:N) - opt(N+1:2*N);
FD.b= opt(end);   
%threshold might be inappropriate

