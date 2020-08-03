function [W,it]= phamDiag(M, maxIter, thresh)
%A= phamDiag(M, <maxIter=100, thresh=1e-5>)
%
% approximate joint diagonalization of positive definite 
% hermitian matrices using general linear transforms
%
% IN:  M       - M(:,:,k) is k-th hermitian(!) input matrix
%      maxIter - maximal number of iterations
%      thresh  - iterations stop when the sqrt of the decrease of the
%                diagonalization criterion function is below this value
%
% OUT: A       - estimated matrix (usually not orthogonal!)
%                (-> inv(A)*M(:,:,k)*inv(A)' is approx. diagonal)
%
% Algorithm & code (jadiag)
%   Dinh-Tuan Pham

% Calling code
%   Benjamin Blankertz, 6/2000, blanker@first.gmd.de

if nargin<2 | isempty(maxIter), maxIter= 100; end
if nargin<3, thresh= 1e-5; end

[N,N,K]= size(M);
C= reshape(M,N,N*K);
W= eye(N);
logdet= 1;
it=0; d= inf;
while it<maxIter & sqrt(d)>thresh, it= it+1;
  [C, crit, W, logdet, d]= jadiag(C, W, logdet);
end

%s= warning; warning off
%A= inv(W);
%warning(s);



function [c, crit, a, logdet, decr] = jadiag(c, a, logdet)
% syntaxe       [c, crit, a, logdet, decr] = jadiag(c, a, logdet)

% Performs approximate joint diagonalization of several matrices.
% The matrices to be diagonalised are given in concatenated form by a
% m x n matrix c, n being a multiple of m. They are tranformed to
% more diagonal with an iteration sweep. The function returns the
% transformed matrices in c. The transformation is also appliied (through
% premultipliation only) to the matrix a (default to the identity matrix
% if not provided) and the result is returned. Further the variable logdet
% is added by twice the number of matrices to be diagonalized times the
% logarithm of the determinant of the transformation (logdet default to 0
% if not given) and is returned. Finally crit contains the logarithm of
% the product of the diagonal elements of all the transformed matrices
% minus the new value of logdet.

[m, n] = size(c);
nmat = fix(n/m);
if (n > nmat*m)
  error('argument must be the concatenation of square matrices')
end

if (nargin < 3)
  logdet = 1;
  if (nargin < 2); a = eye(m); end
end

TINY = 100*realmin;		% use for test
det = 1;
decr = 0;
for i = 2:m
  for j=1:i-1
    c1 = c(i,i:m:n);
    c2 = c(j,j:m:n);
    p = mean(c(i,j:m:n)./c1);
    q = mean(c(i,j:m:n)./c2);
    q1 = mean(c1./c2);
    p2 = mean(c2./c1);

    beta = 1 - p2*q1;				% p1 = q2 = 1
    if (q1 <= p2)	     			% p1 = q2 = 1
      alpha = p2*conj(q) - conj(p);		% q2 = 1
      if (abs(alpha) - beta < TINY)		% beta is always <= 0
        beta = -1;
        decr = decr + nmat*p*conj(p)/p2;	% p1 = 1
        gamma = p/p2;
      else
        gamma = - (p*beta + conj(alpha))/p2;	% p1 = 1
        decr = decr + nmat*(p*conj(p) - alpha*conj(alpha)/beta)/p2;
        beta = beta + (conj(p*alpha) - p*alpha)/p2;
      end
    else
      gamma = p*q1 - q;				% p1 = 1
      if (abs(gamma) - beta < TINY)		% beta is always <= 0
        beta = -1;
        decr = decr + nmat*q*conj(q)/q1;	% q2 = 1
      	alpha = conj(q)/q1;
      else
        alpha = - (conj(q)*beta + conj(gamma))/q1;	% q2 = 1
        decr = decr + nmat*(q*conj(q) - gamma*conj(gamma)/beta)/q1;
        beta = beta + (q*conj(gamma) - conj(q)*gamma)/q1;
      end
    end

    D = beta - sqrt(beta^2 - 4*alpha*gamma);
    T = [1 2*gamma/D; 2*alpha/D 1];
    c([i j],:) = T*c([i j],:);
    for k=0:m:n-m
      c(:,[i+k j+k]) = c(:,[i+k j+k])*conj(T');
    end
    a([i j],:) = T*a([i j],:);
    det = det*abs(1 - T(1,2)*T(2,1));
  end
end
logdet = logdet + 2*nmat*log(det);
crit = 1;
for k=1:m:n
  crit = crit*prod(diag(c(:,[k:k-1+m])));
end
crit = log(crit) - logdet;
