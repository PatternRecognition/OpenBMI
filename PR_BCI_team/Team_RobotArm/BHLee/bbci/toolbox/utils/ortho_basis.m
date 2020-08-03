function A = ortho_basis(A)
% Q = ortho_basis(A)
% 
% Given some orthonormal column vectors in the matrix A, this function
% returns an orthogonal basis which starts with the vectors in A.
% 
% If A's columns are not orthonormal, they are normalized one by one.
%
% ATTENTION: assumes full column rank of A; 
%            assumes that the last unit vectors are l.i. of A's columns.
%

%kraulem 08/05
[n1,n2] = size(A);
if n1<n2
  % more rows than columns.
  error('Assumptions are not met!');
end

A = [A,[zeros(n2,n1-n2);eye(n1-n2)]];
if rank(A)<n1
  error('Assumptions are not met!');
end

for col = 2:n1
  A = col_normalize(A,col);
end

return

function A = col_normalize(A,col)
% take one column (assume the preceding columns to be orthonormal),
% and make it the next orthonormal vector.

% dot product of A(:,col) with every preceding vector:
B = repmat(A(:,col),1,col-1);
B = repmat(sum(A(:,1:col-1).*B,1),size(A,1),1);

% B contains the weights. Subtract this.
A(:,col) = A(:,col)-sum(A(:,1:col-1).*B,2);
A(:,col) = A(:,col)/norm(A(:,col));
return