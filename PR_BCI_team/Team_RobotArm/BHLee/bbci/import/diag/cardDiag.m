function [V, A]= cardDiag(A, seuil, maxIter)
%V= cardDiag(M, <thresh=1e-8, maxIter>);
%
% approximate joint diagonalization of real-valued 
% symmetric matrices using orthogonal transforms
%
% IN:  M       - M(:,:,k) is k-th symmetric input matrix
%      maxIter - maximal number of iterations
%      thresh  - iterations stop when the atan of the angles of
%                all Jacobi rotations are below this value
%
% OUT: V       - estimated orthogonal matrix
%                (-> V'*M(:,:,k)*V is approx. diagonal)
%
% Algorithm & code
%   Jean-Francois Cardoso
%
% for further information see the original file joint_diag.m
% available at http://sig.enst.fr/~cardoso
% [adapted for N-dim array 
%   Benjamin Blankertz, 2/2000, blanker@first.gmd.de]

if nargin<2 | isempty(seuil), seuil=1e-8; end
if nargin<3, maxIter=100; end
[m,m,n] = size(A);

V= eye(m);
it= 0; encore= 1;
while it<maxIter & encore, encore=0; it= it+1;
  for p=1:m-1
    for q=p+1:m
      g    = reshape([A(p,p,:)-A(q,q,:); A(p,q,:)+A(q,p,:)], 2, n);
      gg   = g*g';
      ton  = gg(1,1)-gg(2,2); 
      toff = gg(1,2)+gg(2,1);
      theta= atan2( toff, ton+sqrt(ton*ton+toff*toff) ) / 2;

      if abs(theta) > seuil, encore = 1;
        c= cos(theta); 
        s= sin(theta);

        pair       = [p;q];
        V(:,pair)  = V(:,pair)*[c -s; s c];
        A(pair,:,:)= [ c*A(p,:,:)+s*A(q,:,:); -s*A(p,:,:)+c*A(q,:,:) ];
        A(:,pair,:)= [ c*A(:,p,:)+s*A(:,q,:)  -s*A(:,p,:)+c*A(:,q,:) ];
      end

    end
  end
end
