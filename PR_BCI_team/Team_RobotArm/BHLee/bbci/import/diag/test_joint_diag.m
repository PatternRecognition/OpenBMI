% Calling  the joint approximate diagonalization function.

m=5   % dimension
n=3   % number of matrices



seuil	= 1.0e-12; % precision on joint diag 

compteur=0;


while 1 ; compteur=compteur+1;


% drawing a `random' unitary matrix
U= randn(m)+i*randn(m) ; [U,to_waste]=eig(U+U'); 

% Drawing a random set of commuting matrices
A=zeros(m,m*n);
for imat=1:n
  cols		= 1+(imat-1)*m:imat*m;
  A(:,cols)	= U*diag(randn(m,1)+i*randn(m,1))*U';
end;


% Perturbation of the joint structure ?
% A = A + 0.001*randn(m,m*n);


%% Do it
[ V , DD ] = joint_diag(A,seuil);

%% should be permutation matrix 
abs(V'*U)



end
