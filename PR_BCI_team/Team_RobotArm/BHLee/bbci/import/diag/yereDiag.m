function A= yereDiag(M, A, maxIter, thresh)
%A= yereDiag(M, <A, maxIter=50, thresh=1e-8>);
%
% approximate joint diagonalization of hermitian matrices
% using general linear transforms
%
% IN:  M       - M(:,:,k) is k-th hermitian(!) input matrix
%      A       - initial estimate (default: diag'lizer. of M_1 and M_2)
%      maxIter - number of AC-DC iterations
%      thresh  - iterations stop when frobenius norm of differences
%                between two consecutive A estimates is below this value
%
% OUT: A       - estimated matrix (usually not orthogonal!)
%                (-> inv(A)*M(:,:,k)*inv(A)' is approx. diagonal)
%
% Algorithm
%   Arie Yeredor, Proceedings ICA 2000, p. 33

% Code
%   Benjamin Blankertz, 6/2000, blanker@first.gmd.de

if nargin<2 | isempty(A), [A,D]= eig(M(:,:,2)*M(:,:,1)^-1); end
if nargin<3 | isempty(maxIter), maxIter= 50; end
if nargin<4, thresh= 1e-8; end

N= size(M, 1);

it=0; c=inf;
while it<maxIter & c>thresh, it=it+1;
  Aold= A;
  La= DCphase(M, A);
  for l=randperm(N)
    A= ACphase(M, La, A, l);
  end
  c= norm(A-Aold, 'fro')/N;
end
fprintf('%d iterations -> d= %g\n', it, c);

function La= DCphase(M, A)
[N,N,K]= size(M);

A2= A'*A;
G= inv( conj(A2) .* A2 );

La= zeros(N,K);
for k=1:K
  La(:,k)= G*diag(A'*M(:,:,k)*A);
end



function A= ACphase(M, La, A, l, w)

[N,N,K]= size(M);
if nargin<5, w= ones(K,1); end

P= zeros(N,N);
for k=1:K
%  B= zeros(N,N);
%  for n=setxor(1:N,l)
%    B= B + La(n,k)*A(:,n)*A(:,n)';
%  end
  B= A*diag(La(:,k))*A' - La(l,k)*A(:,l)*A(:,l)';
  P= P  +  w(k)*La(l,k) * (M(:,:,k)-B);
end

[V,D]= eig(P);
[mu,si]= max(diag(D));
if mu<0
  A(:,l)= zeros(N,1);
else
  al= V(:,si)/norm(V(:,si));
  fi= find(al~=0);
  if ~isempty(fi), al= al*sign(al(fi(1))); end
  A(:,l)= al*sqrt(mu/(La(l,:).^2*w));
end















