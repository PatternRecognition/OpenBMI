function S = sparseblock(i, j, A, m, n)
% SPARSEBLOCK - Expand small matrix to large sparse matrix
%
%   S = SPARSEBLOCK(I, J, A, M, N)
%   Returns a sparse matrix S of size [M N] with entries S(I,J)==A.
%   S = SPARSEBLOCK(I, J, A)
%   As above, with M=MAX(I) and N=MAX(J).
%
%   Alternatively, I and J may be logical index vectors.
%   
%   See also SPARSE
%

% Copyright Fraunhofer FIRST.IDA (2004)
% Anton Schwaighofer
% $Id: sparseblock.m,v 1.1 2004/08/17 12:49:16 neuro_cvs Exp $

error(nargchk(3, 5, nargin));

if nargin<4,
  if islogical(i),
    m = length(i);
  else
    m = max(i);
  end
end
if nargin<5,
  if islogical(j),
    n = length(j);
  else
    n = max(max(j), m);
  end
end
S = spalloc(m, n, nnz(A));
S(i,j) = A;
