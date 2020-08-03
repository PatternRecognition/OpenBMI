function [V, D]= j_diag(A, seuil)
%[V, D]= j_diag(A, seuil)
%
% joint diagonalization for real matrices, Jean-Francois Cardoso

% for further information see the original file joint_diag.m
% available at http://sig.enst.fr/~cardoso

if nargin<2, seuil=1e-8; end

[m,nm] = size(A);

V= eye(m);
encore= 1;
while encore, encore=0;   
  for p=1:m-1, Ip= p:m:nm;
    for q=p+1:m, Iq= q:m:nm;

      g    = [ A(p,Ip)-A(q,Iq) ; A(p,Iq)+A(q,Ip) ];
      gg   = g*g';
      ton  = gg(1,1)-gg(2,2); 
      toff = gg(1,2)+gg(2,1);
      theta= 0.5*atan2( toff , ton+sqrt(ton*ton+toff*toff) );

      if abs(theta) > seuil, encore = 1;
        c= cos(theta); 
        s= sin(theta);
        G= [ c -s ; s c ];

        pair        = [p;q];
        V(:,pair)   = V(:,pair)*G;
        A(pair,:)   = G' * A(pair,:);
        A(:,[Ip Iq])= [ c*A(:,Ip)+s*A(:,Iq) -s*A(:,Ip)+c*A(:,Iq) ];
      end

    end
  end
end

D= A;
