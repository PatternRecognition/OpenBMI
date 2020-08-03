function [V,C,moni]=fastDiag(Co,V,itr,thres,stiefel);
%ultra fast and efficient
%simultaneous diagonalization of a set of matrices
%by an general linear transformation
%with a linear least squares method 
%operated in a mutiplicative update procedure 
%Usage:
% [V,C,moni]=lsdiag2(Co,V,itr,thres,stiefel);
Anneal=1;
Stepsize=0.1;
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

% Scale W by power of 2 so that its norm is <1 .
%neseccary to make approximation assumptions hold
%i.e W should be small
if 0
    [f,e] = log2(norm(W,'inf'));
   % s = max(0,e/2);
   
    s = max(0,e-1);
    W = W/(2^s );
    EN(t)=(2^s );
%keyboard
end

EN(t)=1;
if 0
  s=Stepsize   / norm(W, 'inf');
 W = W * s;
 EN(t)=s;
end

%W=W/(1+trace(mean(C,3)));
%EN(t)=norm(eye(N)+W,'fro');
EW(t)=norm(W);


V=expm(W)*V;  %multiplicative update

%V=(eye(N)+W)*V;


V=diag(1./sqrt(diag(V*V')))*V;  %norm(V)=1 
%V=inv(normit(inv(V)));

%V=(eye(N)+W + W*W/2  + W^3/6 +W^4/24)*V;
%V=(eye(N)+W + W*W/2  + W^3/6 +W^4/24 +W^5/120)*V;

% update correlation matrices 
for k=1:K
    C(:,:,k)=V*Co(:,:,k)*V';
end 
  
EE(t)=cost_off(C);

%fprintf('iter: %d    rho: %2.5e   off: %.3e\n',t,EW(t),EE(t))

if abs(EW(t)-EW(t-1))<thres,
    break
end 

end
moni.etime=toc;
moni.normW=EW;
moni.normWW=EN;
moni.errdiag=EE;
moni.iter=t;
moni.V=V;



return




function [cost]=cost_off(C,V)
% diagonalization error
%  C set of matrices
%  V diagonalizing matrix

[N,N,K]=size(C);

if nargin>1,
  for k=1:K,
    C(:,:,k)=V*C(:,:,k)*V';
  end
end



cost=0;
for k=1:K
  Ck=C(:,:,k);Ck=Ck(:);
  Ck(1:N+1:N*N)=0;
  cost=cost+norm(Ck)^2;
end








