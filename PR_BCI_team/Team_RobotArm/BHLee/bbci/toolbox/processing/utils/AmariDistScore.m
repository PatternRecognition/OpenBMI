function m= AmariDistScore(C,A)
% m= AmariDistScore(C,A)
%
% computes a pseudo-distance between matrices A and C invariant to scaling and permutation
% returns the Frobenius norm of the off-diagonal elements of inv(C)*A
% normalized by number of sources
%
%  Input: square matrices C and A of same size
%
%  Output: a scalar value m>=0 which is zero if the matrix  is identical to A up to scaling and permutation
%
%
% code by Benjamin Blankertz 2000
% maintained by Andreas Ziehe  for NeuroToolbox

[N,N]=size(C);
W=inv(C);
m= norm( eyeLike(W*A)-eye(size(W)), 'fro' )./(N-1);



function [eyelike, S, P]= eyeLike(E)
%[eyelike, S, P]= eyeLike(E);
%
% calculates a permutation matrix P and a scaling (diagonal) maxtrix S
% such that S*P*E is eyelike (so permutation acts on the rows of E).
% E must be a square matrix.

[N, N]= size(E);

R= E./repmat(sum(abs(E),2),1,N);
P= getBigDiag(R);
S= diag(1./diag(P*E));

eyelike= S*P*E;


function [P, S]= getBigDiag(A)
%[P, S]= getBigDiag(A)
%
% calculates a permutation matrix P and a sign switching matrix S such that 
% S*P*A has big elements on the diagonal (so permutation acts on the rows of A).
% A must be a square matrix

[N, N]= size(A);
Aabs= abs(A);

P= zeros(N);
for n=1:N
  [dum, ind]= sort(-Aabs(:));
  chosenH= ceil(ind(1)/N);
  chosenV= ind(1)-N*(chosenH-1);
  P(chosenH,chosenV)= 1;
  Aabs(chosenV,:)= repmat(-inf, 1, N);
  Aabs(:,chosenH)= repmat(-inf, N, 1);
end

S= diag(sign(diag(P*A)));






