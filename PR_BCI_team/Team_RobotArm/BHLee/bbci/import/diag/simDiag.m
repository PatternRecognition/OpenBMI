function [V, moni] = simDiag(M,method,maxiter)
%wrapper for various methods of joint diagonalization 
%of a set of matrices
%
%uses code by 
% Jean Francoise Cardoso
% Dhin-Tuan Pham 
% Arie Yeredor
% Andreas Ziehe
%
%Usage:
%function [V, moni] = simDiag(Cx,method,params)
%IN
% M       set of matrices in an 3d-array M(:,:,k)
% method   string with method name 
%          currently implemented: ja_diag
%params    parameter specification
%  
%OUT
%  V     joint diagonalizer  (a square matrix)
%  moni   various monitoring variables, stored as an object
%         e.g. moni.errdiag  diagonalization error
%                  .etime    running time in tic-toc seconds
%                  .diags    approximately diagonal matrices
%                            containing the eigenvalues 
%                  .iter    number of iterations
%                  .V        common eigenvectors
  
 
%get dimensions  
[N,NN,K]= size(M);

if N~=NN,
  error('Matrices must be square')
end

%shortcut if only one matrix needs to be diagonalized
%if K==1,
%  V=eig(squeeze(M));
%end

%initial value
V0= eye(N);

%stopping criteria
threshold=sqrt(eps);

if nargin<3,
%maximum number of iterations
maxiter=200;
end

switch lower(method)
 
 case {'jadiag','pham','phamdiag'}
   %convert to 2d array 
   M=reshape(M,N,N*K);
   tic;
   [V, CC,EE] = jadiag_int(M,V0,threshold,maxiter);
   moni.etime=toc;
   moni.iter=length(EE);
   moni.errdiag=EE;
   moni.method='jadiag';
   
 case {'acdc','yeredor','yerediag'}
   tic;
   [ V, qDs,Nit,Cls]=acdc(M,ones(1,K),V0,[],maxiter,threshold);
   V=inv(V);
   moni.etime=toc;
   moni.qDs=qDs;
   moni.iter=Nit;
   moni.errdiag=Cls;
   moni.res=cost_off(M,V);
   moni.method='acdc';
   
 case {'joint_diag','cardoso','jade','carddiag','jacobi'}
  %convert to 2d array
  M=reshape(M,N,N*K);
  tic;
  [ V ,  qDs,EE ]= rjd(M,threshold);
  V=V';
  moni.etime=toc;
  moni.qDs=qDs;
  moni.V=V;
  moni.iter=length(EE);
  moni.errdiag=EE;
  moni.method='jacobi';
 case {'fastdiag','nolte','fast'}
  tic;
  [V,CC,moni]=fastDiag(M,V0,maxiter,threshold);
  moni.etime=toc;
  moni.method='fastDiag';
  
  case {'fasto','stiefel'}
  tic;
   [V,CC,moni]=fastoDiag(M,V0,maxiter,threshold);
  moni.etime=toc;
    moni.method='orthogonal fastDiag';
 otherwise
  error('Unknown method.')
end



return

%HERE ARE THE FUNCTIONS FOR JOINT DIAGONALIZATION
%THE COPYRIGHTS REMAIN WITH THE ORIGINAL AUTHORS 

function [ V ,  A, EE ]= rjd(A,threshold)
%***************************************
% joint diagonalization (possibly
% approximate) of REAL matrices.
%***************************************
% This function minimizes a joint diagonality criterion
% through n matrices of size m by m.
%
% Input :
% * the  n by nm matrix A is the concatenation of m matrices
%   with size n by n. We denote A = [ A1 A2 .... An ]
% * threshold is an optional small number (typically = 1.0e-8 see below).
%
% Output :
% * V is a n by n orthogonal matrix.
% * qDs is the concatenation of (quasi)diagonal n by n matrices:
%   qDs = [ D1 D2 ... Dn ] where A1 = V*D1*V' ,..., An =V*Dn*V'.
%
% The algorithm finds an orthogonal matrix V
% such that the matrices D1,...,Dn  are as diagonal as possible,
% providing a kind of `average eigen-structure' shared
% by the matrices A1 ,..., An.
% If the matrices A1,...,An do have an exact common eigen-structure
% ie a common othonormal set eigenvectors, then the algorithm finds it.
% The eigenvectors THEN are the column vectors of V
% and D1, ...,Dn are diagonal matrices.
% 
% The algorithm implements a properly extended Jacobi algorithm.
% The algorithm stops when all the Givens rotations in a sweep
% have sines smaller than 'threshold'.
% In many applications, the notion of approximate joint diagonalization
% is ad hoc and very small values of threshold do not make sense
% because the diagonality criterion itself is ad hoc.
% Hence, it is often not necessary to push the accuracy of
% the rotation matrix V to the machine precision.
% It is defaulted here to the square root of the machine precision.
% 
%
% Author : Jean-Francois Cardoso. cardoso@sig.enst.fr
% This software is for non commercial use only.
% It is freeware but not in the public domain.
% A version for the complex case is available
% upon request at cardoso@sig.enst.fr
%-----------------------------------------------------
% Two References:
%
% The algorithm is explained in:
%
%@article{SC-siam,
%   HTML =	"ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",
%   author = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
%   journal = "{SIAM} J. Mat. Anal. Appl.",
%   title = "Jacobi angles for simultaneous diagonalization",
%   pages = "161--164",
%   volume = "17",
%   number = "1",
%   month = jan,
%   year = {1995}}
%
%  The perturbation analysis is described in
%
%@techreport{PertDJ,
%   author = "{J.F. Cardoso}",
%   HTML =	"ftp://sig.enst.fr/pub/jfc/Papers/joint_diag_pert_an.ps",
%   institution = "T\'{e}l\'{e}com {P}aris",
%   title = "Perturbation of joint diagonalizers. Ref\# 94D027",
%   year = "1994" }
%
%
%
[m,nm] = size(A);
V=eye(m);

if nargin==1, threshold=sqrt(eps); end;
it=2;
encore=1;
EE(1)=cost_off(reshape(A,m,m,nm/m));
while encore,  encore=0;
 for p=1:m-1,
  for q=p+1:m,
   %%%computation of Givens rotations
   g=[ A(p,p:m:nm)-A(q,q:m:nm) ; A(p,q:m:nm)+A(q,p:m:nm) ];
   g=g*g';
   ton =g(1,1)-g(2,2); toff=g(1,2)+g(2,1);
   theta=0.5*atan2( toff , ton+sqrt(ton*ton+toff*toff) );
   c=cos(theta);s=sin(theta);
   encore=encore | (abs(s)>threshold);
    %%%update of the A and V matrices 
   if (abs(s)>threshold) ,
    Mp=A(:,p:m:nm);Mq=A(:,q:m:nm);
    A(:,p:m:nm)=c*Mp+s*Mq;A(:,q:m:nm)=c*Mq-s*Mp;
    rowp=A(p,:);rowq=A(q,:);
    A(p,:)=c*rowp+s*rowq;A(q,:)=c*rowq-s*rowp;
    temp=V(:,p);V(:,p)=c*V(:,p)+s*V(:,q); V(:,q)=c*V(:,q)-s*temp;
   end%%of the if
  end%%of the loop on q
 end%%of the loop on p
 
 EE(it)=cost_off(reshape(A,m,m,nm/m));
 it=it+1;
end%%of the while loop
%qDs = A ;
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PHAMS METHOD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a, c,EE] = jadiag_int(c, a, eps, maxiter)
% syntaxe       [a, c,EE] = jadiag(c, a, eps, maxiter)

% Performs approximate joint diagonalization of several matrices.
% The matrices to be diagonalised are given in concatenated form by a
% m x n matrix c, n being a multiple of m. They are transformed to nearly
% diagonal and the transformation is also applied to a. Thus a yields the
% diagonalising matrix if it is initialised by the identity matrix (the
% default), otherwise it is the product of the diagonalsing matrix
% with its initialized value.
% The stoping criterion is that the square norm (with respect to a
% certain metric) of the relative "gradient" is less than eps or the
% number of step attains maxstep. The above squared norm also equals
% approximatively the decrease of the criterion at this step.
% eps defaults to m*(m-1)*1e-4, maxiter to 15 and a to the identity matrix

[m, n] = size(c);
nmat = fix(n/m);
if (n > nmat*m)
  error('argument must be the concatenation of square matrices')
end

if (nargin < 4); maxiter = 15; end
if (nargin < 3); eps = m*(m-1)*1e-6; end
if (nargin < 2); a = eye(m); end

one = 1 + 10e-12;			% considered as equal to 1
EE(1)=cost_off(reshape(c,m,m,nmat));
for it = 1:maxiter
  decr = 0;
  for i = 2:m
    for j=1:i-1
      c1 = c(i,i:m:n);
      c2 = c(j,j:m:n);
      g12 = mean(c(i,j:m:n)./c1);	% this is g_{ij}
      g21 = mean(c(i,j:m:n)./c2);	% this is the conjugate of g_{ji}
      omega21 = mean(c1./c2);
      omega12 = mean(c2./c1);
      omega = sqrt(omega12*omega21);
      tmp = sqrt(omega21/omega12);
      tmp1 = (tmp*g12 + g21)/(omega + 1);
      omega = max(omega, one);
      tmp2 = (tmp*g12 - g21)/(omega - 1);
      h12 = tmp1 + tmp2;			% this is twice h_{ij}
      h21 = conj((tmp1 - tmp2)/tmp);		% this is twice h_{ji}
      decr = decr + nmat*(g12*conj(h12) + g21*h21)/2;

      tmp = 1 + 0.5i*imag(h12*h21);	% = 1 + (h12*h21 - conj(h12*h21))/4
      T = eye(2) - [0 h12; h21 0]/(tmp + sqrt(tmp^2 - h12*h21));
      c([i j],:) = T*c([i j],:);
      for k=0:m:n-m
        c(:,[i+k j+k]) = c(:,[i+k j+k])*T';
      end
      a([i j],:) = T*a([i j],:);
    end
  end
EE(it+1)=cost_off(reshape(c,m,m,nmat));
 % fprintf('iteration %d, gradient norm %.6g\n', it, sqrt(decr/(m*(m+1))));
  if decr < eps; break; end		% convergence achieved
end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%non-orthogonal fastDiag
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [V,C,moni]=fastDiag(Co,V,itr,thres);
%ultra fast and efficient
%simultaneous diagonalization of a set of matrices
%by an general linear transformation
%with a linear least squares method 
%operated in a mutiplicative update procedure 
%Usage:
% [V,C,moni]=fastDiag(Co,V,itr,thres);

[N,M,K]=size(Co);
if K<2
    error('at least two input matrices are required');
end
% choose clever starting point, if V is not given

if ~exist('V') |isempty(V),
   [V D0]=eig(mean(Co,3));
   V=V';
  %   M0= Co(:,:,1)*Co(:,:,2)^-1;
  %   [V,D0]= eig(M0);
  %   V=inv(V);
end


if ~exist('thres') | isempty(thres),
    thres=10e-8;
end

if ~exist('itr') |isempty(itr),
    itr=2*N;
end

%initialize
tic;
%I=speye(N); %Identity matrix
%W=V-diag(diag(V));

for k=1:K
  C(:,:,k)=(V*Co(:,:,k)*V');
end
EE(1)=cost_off(C);
%main iteration
for t=2:itr,
  
  W=zeros(N,N);  
  
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
  

  % Scale W by power of 2 so that its norm is <1 .
  %neseccary to make approximation assumptions hold
  %i.e W should be small
  
  %if 0,
  [f,e] = log2(norm(W,'inf'));
  % s = max(0,e/2);

  s = max(0,e-1);
  W = W/(2^s );
  %   EN(t)=(2^s );
  
  %end
  
  EW(t)=norm(W);
  %multiplicative update
  V=(eye(N)+W)*V;
  
  %re-normalization
  V=diag(1./sqrt(diag(V*V')))*V;  %norm(V)=1 
  
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

return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [V,C,moni]=fastoDiag(Co,V,itr,thres);
%orthogonal diagonalization
%with a linear least squares method 
%operated in a mutiplicative update procedure 
%Usage:
% [V,C,moni]=fastoDiag(Co,V,itr,thres);

[N,M,K]=size(Co);

if ~exist('thres') | isempty(thres),
    thres=10e-12;
end

if ~exist('itr') |isempty(itr),
    itr=2*N;
end

%initialize
tic;
%I=speye(N); %Identity matrix
%W=V-diag(diag(V));

for k=1:K
  C(:,:,k)=(V*Co(:,:,k)*V');
end
EE(1)=cost_off(C);
%main iteration
for t=2:itr,
 
  W=zeros(N,N);  
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
  W=(W'-W); %constrain W to be  anti-symmetric

  % Scale W by power of 2 so that its norm is <1 .
  %neseccary to make approximation assumptions hold
  %i.e W should be small
  %if 0
    [f,e] = log2(norm(W,'inf'));
    % s = max(0,e/2);
    
    s = max(0,e-1);
    W = W/(2^s );
    %end
    
        
    EW(t)=norm(W);


    V=expm(W)*V;  %multiplicative update

  %  V=diag(1./sqrt(diag(V*V')))*V;  %norm(V)=1 

    % update correlation matrices 
    for k=1:K
      C(:,:,k)=V*Co(:,:,k)*V';
    end 
    
    EE(t)=cost_off(C);

    %fprintf('iter: %d    rho: %2.5e   off: %.3e\n',t,EW(t),EE(t))

    if abs(EE(t)-EE(t-1))<thres,
      break
    end 

end
moni.etime=toc;
moni.normW=EW;
moni.errdiag=EE;
moni.iter=t;
moni.V=V;



return



function [A,Lam,Nit,Cls]=acdc(M,w,A0,Lam0,MAXIT,TOL);

%acdc: appoximate joint diagonalization
%(in the direct Least-Squares sense) of 
%a set of Hermitian matrices, using the
%iterative AC-DC algorithm.
%
%the basic call:
%[A,Lam]=acdc(M);
%%
%[A,Lam,Nit,Cls]=
%	acdc(M,w,A0,Lam0);
%
%(additional) Inputs:
%%
%TOL - a tolerance value on the change of
%      C_{LS}. AC-DC stops when the
%      decrease of C_{LS} is below tol.
%      Originally set to:
%            10^-3/(N*N*sum(w));
%
%MAXIT - maximum number of allowed full
%        iterations.
%        Originally set to: 50;
%
%INTLC - number of AC sweeps to interlace
%        dc sweeps.
%        Originally set to: 1.
%
%-----------------------------------------
%
%Note that the implementation here is
%somewhat wasteful (computationally),
%mainly in performing a full eigenvalue
%decomposition at each AC iteration, 
%where in fact only the largest eigenvalue
%(and associated eigenvector) are needed,
%and could be extracted e.g. using the 
%power method. However, for small N (<10),
%the matlab eig function runs faster than
%the power method, so we stick to it.

%-----------------------------------------
%version R1.0, June 2000.
%By Arie Yeredor  arie@eng.tau.ac.il
%
%rev. R1.1, December 2001
%forced s=real(diag(S)) rather than just s=diag(S)
%in the AC phase. S is always real anyway; however,
%it may be set to a complex number with a zero 
%imaginary part, in which case the following
%max operation yields the max abs value, rather
%than the true max. This fixes that problem. -AY
%
%Permission is granted to use and 
%distribute this code unaltered. You may 
%also alter it for your own needs, but you
%may not distribute the altered code 
%without obtaining the author's explicit
%consent.
%comments, bug reports, questions 
%and suggestions are welcome.
%
%References:
%[1] Yeredor, A., Approximate Joint 
%Diagonalization Using Non-Orthogonal
%Matrices, Proceedings of ICA2000, 
%pp.33-38, Helsinki, June 2000.
%[2] Yeredor, A., Non-Orthogonal Joint 
%Diagonalization in the Least-Squares 
%Sense with Application in Blind Source
%Separation, IEEE Trans. On Signal Processing,
%vol. 50 no. 7 pp. 1545-1553, July 2002.


[N N1 K]=size(M);
if N~=N1
    error('input matrices must be square');
end
if K<2
    error('at least two input matrices are required');
end

if exist('w','var') & ~isempty(w)
    w=w(:);
    if length(w)~=K
        error('length of w must equal K')
    end   
    if any(w<0)
        error('all weights must be positive');
    end
else
    w=ones(K,1);
end

if exist('A0','var') & ~isempty(A0)
    [NA0,Nc]=size(A0);
    if NA0~=N
        error('A0 must have the same number of rows as the target matrices')
    end
else
    A0=eye(N);
    Nc=N;
end

if exist('Lam0','var') & ~isempty(Lam0)
    [NL0,KL0]=size(Lam0);
    if NL0~=Nc
        error('each vector in Lam0 must have M elements')
    end
    if KL0~=K
        error('Lam0 must have K vectors')
    end
    if ~isreal(Lam0)
        error('Lam0 must be real')
    end
    skipAC=0;
else
    skipAC=1;
end

%-----------------------------------------
%  here's where the fixed processing-
%  parameters are set (and may be 
%  modified):
%-----------------------------------------
%TOL=1e-3/(N*N*sum(w))
%MAXIT=100;
INTLC=5;

%-----------------------------------------
%  and this is where we start working
%-----------------------------------------

Cls=zeros(MAXIT,1);
Lam=zeros(N,K);
A=A0;
for Nit=1:MAXIT
    
    if ~skipAC
        
        %AC phase   
        for nsw=1:INTLC
            for l=1:Nc
                P=zeros(N);
                for k=1:K
                    D=M(:,:,k);
                    for nc=[1:l-1 l+1:Nc]
                        a=A(:,nc);
                        D=D-Lam(nc,k)*a*a';
                    end
                    P=P+w(k)*Lam(l,k)*D;
                end
                [V S]=eig(P);
                s=real(diag(S));     %R1.1 - ay
                [vix,mix]=max(s);
                if vix>0
                    al=V(:,mix);
                    %this makes sure the 1st nonzero
                    %element is positive, to avoid
                    %hopping between sign changes:
                    fnz=find(al~=0);
                    al=al*sign(al(fnz(1)));
                    lam=Lam(l,:);
                    f=vix/((lam.*lam)*w);
                    a=al*sqrt(f);
                else
                    a=zeros(N,1);
                end	
                A(:,l)=a;
            end	%sweep
        end		%interlaces
    end			%skip AC
    skipAC=0;
    
    %DC phase
    AtA=A'*A;
    AtA2=AtA.*conj(AtA);
    G=inv(AtA2);
    for k=1:K
        Lam(:,k)=G*diag(A'*M(:,:,k)*A);
        L=diag(Lam(:,k));
        D=M(:,:,k)-A*L*A';
        Cls(Nit)=Cls(Nit)+w(k)*sum(sum(D.*conj(D)));
    end
   % EE(Nit)=cost_off(M,A);
    if Nit>1
        if abs(Cls(Nit)-Cls(Nit-1))<TOL
            break
        end
    end
    
end
Cls=Cls(1:Nit);

return