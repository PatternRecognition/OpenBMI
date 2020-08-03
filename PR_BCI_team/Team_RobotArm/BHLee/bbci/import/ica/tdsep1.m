function W= tdsep1(x, tau)
%W= tdsep1(x, <tau=0:1>)
%
% computes ICA demixing matrix, Algo: A.Ziehe
% takes signals to be the rows of x

if nargin<2, tau=[0 1]; end  

[N, T]= size(x);
x= x - mean(x,2)*ones(1,T);          %% Ex = 0
SPH= inv(sqrtm(cor2(x',tau(1))));
spx= SPH*x;

nTaus= length(tau);
if nTaus==2,                         %% for two matrices, solve directly 
  M= cor2(spx', tau(2));             %% as general eigenvalue problem
  [Q,D]= eig(M);
else
  M= [];
  for t= tau,
    M= [M cor2(spx', t)];
  end
  [Q,d]= jdiag(M, 1e-8);
end

W= Q'*SPH;                           %% compute demixing matrix
