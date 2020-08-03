function [V,C,moni]=lsdiag_ls(Co,V,itr,thres,stiefel);
%ultra fast and efficient
%simultaneous diagonalization of a set of matrices
%by an general linear transformation
%with a linear least squares method 
%operated in a mutiplicative update procedure 
%Usage:
% [V,C,moni]=lsdiag2(Co,V,itr,thres,stiefel);

[N,M,K]=size(Co);

% choose clever starting point, if V is not given

if ~exist('V') |isempty(V),
   [V D0]=eig(mean(Co,3));
   V=V';
  %   M0= Co(:,:,1)*Co(:,:,2)^-1;
  %   [V,D0]= eig(M0);
  %   V=inv(V);
end

 
if nargin<5,
  stiefel=0;  % 1 means work on Stiefel manifold for orthognal diagonalizers
end

if ~exist('thres') | isempty(thres),
    thres=10e-12;
end

if ~exist('itr') |isempty(itr),
    itr=2*N;
end

if K==1,
   %fprintf('Diagonalizing only one matrix...')
   stiefel=1;
end

%initialize

tic;
%I=speye(N); %Identity matrix
%W=V-diag(diag(V));

for k=1:K
  C(:,:,k)=(V*Co(:,:,k)*V');
end
 cost_old=cost_off(C);
EE(1)=cost_old;
 %main iteration
for t=2:itr,
  
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
  W=(W'-W); %W is anti-symmetric
 
  else
    %non-orthogonal case
   
   for n=1:N,
    for m=1:M,
      Y(n,m)=sum(((C(m,n,:)+C(n,m,:))/2) .*C(m,m,:));  %nicht symmetrisch
      Z(n,m)=sum(C(m,m,:).*C(n,n,:));
    end
   end
   
   for n=1:N,
     for m=1:M,
       if n<m,
	 Determ=(Z(n,n)*Z(m,m))-Z(m,n)^2; %Determinante
	 W(n,m)=-1/(Determ)*(Z(n,n).*Y(n,m)-Z(m,n).*Y(m,n)); 
	 W(m,n)=-1/(Determ)*(Z(m,m).*Y(m,n)-Z(n,m).*Y(n,m));
       end       
     end
   end
   %fprintf('.')
  end          
  





 %do some line search
 lam=1;
 lamfactor=0.1;
 
 for i_lin=1:30 %100,
   

   

   % update correlation matrices 
   for k=1:K
     Ctmp(:,:,k)=V*Co(:,:,k)*V';
   end

   cost=cost_off(Ctmp);
	 if (cost<cost_old) | (cost<10e-8),  %  | (sum(W(:))<10e-7),
	   cost_old=cost;
          fprintf('.')
	   %use the found solition, because its good
	   C=Ctmp;
	   %update V
	   %V=(Id+W)*V;
	   V=W*V+V;
	  % V=inv(normit(inv(V))); 
	   break 
	 else
	  % cost_old=cost;
	   W=lamfactor*W;
	   lam=lamfactor*lam;
	%   fprintf('+');
	   
	 end	   
	 if i_lin==30,
	    fprintf('+');
	 %    fprintf('+ (%0.3e) ', WY);
	 %   keyboard
	 
	 %%
	 
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
 
	 end
	 
       end 
      fprintf(' m= %d n= %d  lam=%.2e  ||W||=%.2e  off:%.3e \n',m,n,lam,norm(W(:)) , cost);  
		 

 % V=(eye(N)+W)*V;  %multiplicative update
   
   
EE(t)=cost_off(C);

fprintf('iter: %d      off: %.3e\n',t,EE(t))

%if abs(EW(t)-EW(t-1))<thres,
%  if abs(EE(t)-EE(t-1))<thres,
%    break
%    
%end

end
moni.errdiag=EE;
moni.etime=toc;
moni.normW=EW;
moni.lam=EL;
moni.iter=t;
moni.V=V;


 abs(EE(t)-EE(t-1))









