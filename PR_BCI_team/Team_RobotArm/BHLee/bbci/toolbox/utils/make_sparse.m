function S= make_sparse(S)
%S= make_sparse(A)

[i,j,s] = find(S);
[m,n] = size(S);
S = sparse(i,j,s,m,n);
