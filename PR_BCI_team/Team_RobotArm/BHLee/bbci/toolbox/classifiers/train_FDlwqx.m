function FD= train_FDlwqx(xTr, yTr, C)
%FD= train_FDlwqx(xTr, yTr, C)
%
% mathematical programming of Fisher discriminant,
% linear w - quadratic xi minimization (SMika)
%
%   min. ||w||_1 + C*mean xi_k^2   s.t.  ....
%
%   w'*xTr(:,k) + b =  1-xi(k)  for class 1 (yTr(k)==1)
%                   = -1+xi(k)  for class 2 (yTr(k)==-1)
%
% C can be an element between 0 and infinity
%
% GLOBZ  LPENV

start_cplex;

if size(yTr,1)==2, yTr= [-1 1]*yTr; end
[N,K]= size(xTr);

% [w+, w-, xi, b]
Q  = spalloc(2*N+K+1, 2*N+K+1, K);
Q(2*N+1:2*N+K, 2*N+1:2*N+K) = C/K*speye(K);
c  = [ones(2*N,1); zeros(K,1); 0];
lb = [zeros(2*N,1); -inf*ones(K+1,1)];
ub = inf*ones(2*N+K+1, 1);

A  = sparse([xTr', -xTr', -speye(K), ones(K,1)]);
a= (yTr==1)/sum(yTr==1)+(yTr==-1)/sum(yTr==-1);
a=a'*K;
neq= K;

[res, lambda, how]= qp_solve(LPENV, Q, c, A, a, lb, ub, neq, 0);
if ~isequal(how,'OK'),
  fprintf(2, 'NOT OKI %s', how);
end
  
FD.w= res(1:N) - res(N+1:2*N);
FD.b= -mean(xTr,2)'*FD.w;   
%threshold might be inappropriate
