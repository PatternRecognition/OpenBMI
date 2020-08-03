function [V,C,moni]=lsdiag_heu(Co,V,itr,stiefel);
%simultaneous diagonalization for BSS (x=As;)
%by an general linear transformation
%wit a linear least squares method
  
[N,M,K]=size(Co);

%initialize

I=speye(N); %Identity matrix
W=V-diag(diag(V));
EL(1)=1;
if nargin<4,
  stiefel=0;  % 1 means work on Stiefel manifold for orthognal diagonalizers
end

if K==1,
  stiefel=1;
end

tic;
for k=1:K
  C(:,:,k)=V*Co(:,:,k)*V';
end

%evaluate errors at beginning
[EN(1,1) EN(2,1)]=err_fun(C,W);
EN(3,1)=errdiag(C);

for t=2:itr
  alph=0;
  W=zeros(N,N);  
  
  if  stiefel==1  ,
  %orthogonal case    
    for n=1:N,
      for m=1:M,
	if n<m,
	  ZZ=sum(C(n,m,:).*(C(n,n,:)-C(m,m,:)));
	  NN=sum((C(m,m,:)-C(n,n,:)).^2);      
	  W(m,n)=ZZ/NN;
	end   

      end
    end
W=(W'-W);
    %    W=W-W';
  % W=W-diag(diag(W));    
  else
    %non-orthogonal case
    
    for n=1:N,
      for m=1:M,
	
	Y(n,m)=sum(C(m,n,:).*C(m,m,:));  %nicht symmetrisch
	Z(n,m)=sum(C(m,m,:).*C(n,n,:));
      end
    end
% if t>10,
%    keyboard
%  end
    for n=1:N,
      for m=1:M,
	
	if n<m,
	 % H=[Z(n,n) Z(m,n); Z(n,m) Z(m,m)];
	 % YY=[ ]
	 % [vvv,ddd]=eig(H);
	 % if any(diag(ddd)<=0),
	 %   alph=1
	 % keyboard
	 % end
	  
	  D(n,m)=(Z(n,n)+alph*Z(n,n))*(Z(m,m)+alph*Z(m,m))-Z(m,n)^2;
	  
	  W(n,m)=-1/(D(n,m))*((Z(n,n)+alph*Z(n,n)).*Y(n,m)-Z(m,n).*Y(m,n)); 
	  W(m,n)=-1/(D(n,m))*((Z(m,m)+alph*Z(m,m)).*Y(m,n)-Z(n,m).*Y(n,m));
	alph=0;
	end       
      
      end
    end
 end          

EW(t)=norm(W,'fro');

%heuristics to help convergence

if  (t>2) & (EW(t)>EW(t-1)) ,%& (EW(t)>10e-5) ,
  lam=0.99*(EW(t-1)/EW(t));
  W=lam*W; 
 
  EW(t)=norm(W,'fro');
  EL(t)=lam;
else
  EL(t)=1;
end
 
 Vnew=I+W;

 if stiefel==1;
   SPH= (Vnew*Vnew')^(-0.5);
   Vnew=SPH*Vnew;
 end

 V=Vnew*V;

 %re-normalization
 %if stiefel~=1,
   V=inv(normit(inv(V)));
 %end

 % update correlation matrices 
 for k=1:K
   C(:,:,k)=V*Co(:,:,k)*V';
 end 
 
%compute diagonalization error
 [EN(1,t) EN(2,t)]=err_fun(C,W);
 EN(3,t)=errdiag(C); 

% monitoring
% fprintf('%d %f %f \n',t,EN(t),alph);

end
 EN(1,:)=EN(1,:)./EN(1,1);
 EN(2,:)=EN(2,:)./EN(2,1);
 EN(3,:)=EN(3,:)./EN(3,1); 
% monitoring
 moni.etime=toc;
 moni.errdiag=EN;
 moni.normW=EW;
 moni.lambda=EL;
 moni.V=V;














