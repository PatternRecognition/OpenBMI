function [V, A]= joidia(A, seuil)
%[V, D]= joidia(A, <seuil=1e-8>);
%
% joint diagonalization for real matrices, Jean-Francois Cardoso
% (A is a sequence of square matrices given as an MxMxN array)

% adapted for N-dim array by Benjamin Blankertz, Feb 2000
% for further information see the original file joint_diag.m
% available at http://sig.enst.fr/~cardoso

if nargin<2, seuil=1e-8; end
[m,mtoo,n] = size(A);
if m~=mtoo, error('matrices must be square'); end

V= eye(m);
encore= 1;
while encore, encore=0;   
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
