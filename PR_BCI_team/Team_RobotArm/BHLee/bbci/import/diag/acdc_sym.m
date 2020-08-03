function [A,Lam,Nit,Cls]=...
    acdc_sym(M,w,A0,Lam0);

%acdc_sym: appoximate joint diagonalization
%(in the direct Least-Squares sense) of 
%a set of symmetric matrices, using the
%iterative AC-DC algorithm.
%
%the basic call:
%[A,Lam]=acdc(M);
%
%Inputs:
%
%M(N,N,K) - the input set of K NxN 
%           "target matrices". Note that
%           all matrices must be 
%           symmetric (but need not be 
%           positive-definite).
%
%Outputs:
%
%A(N,N)   - the diagonalizing matrix.
%
%Lam(N,K) - the diagonal values of the K
%           diagonal matrices.
%
%The algorithm finds an NxN matrix A and
%K diagonal matrices
%         L(:,:,k)=diag(Lam(:,k))
%such that
% C_{LS}=
% \sum_k\|M(:,:,k)-A*L(:,:,k)*A'\|_F^2
%is minimized.
%
%-----------------------------------------
%   Optional additional input/output
%   parameters:
%-----------------------------------------
%
%[A,Lam,Nit,Cls]=
%	acdc(M,w,A0,Lam0);
%
%(additional) Inputs:
%
%w(K) - a set of positive weights such that
%       C_{LS}=
%       \sum_k\w(k)|M(:,:,k)-A*L(:,:,k)*A'\|_F^2
%       Default: w=ones(K,1);
%
%A0 - an initial guess for A
%     default: eye(N);
%
%Lam0 - an initial guess for the values of
%       Lam. If specified, an AC phase is
%       run first; otherwise, a DC phase is
%       run first.
%
%(additional) Outputs:
%
%Nit - number of full iterations
%
%Cls - vector of Nit Cls values
%
%-----------------------------------------
% Additional fixed processing parameters
%-----------------------------------------
%
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
    skipAC=0;
else
    skipAC=1;
end

%-----------------------------------------
%  here's where the fixed processing-
%  parameters are set (and may be 
%  modified):
%-----------------------------------------
TOL=1e-3/(N*N*sum(w));
MAXIT=500;
INTLC=1;

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
                        D=D-Lam(nc,k)*a*conj(a');
                    end
                    P=P+w(k)*Lam(l,k)*conj(D);
                end
                Pgal=[real(P) -imag(P);-imag(P) -real(P)];
                [V S]=eig(Pgal);
                s=real(diag(S));  %the real is needed to ensure
                %proper max operation!
                [vix,mix]=max(s);
                if vix>0
                    gd=V(:,mix);
                    al=gd(1:N)+1j*gd(N+[1:N]);
                    %this makes sure the 1st nonzero
                    %element's real part is positive, to avoid
                    %hopping between sign changes:
                    fnz=find(real(al)~=0);
                    al=al*sign(real(al(fnz(1))));
                    lam=Lam(l,:);
                    f=vix/((lam.*conj(lam))*w);
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
    AtA2=AtA.*AtA;
    G=inv(AtA2);
    for k=1:K
        Lam(:,k)=G*diag(A'*M(:,:,k)*conj(A));
        L=diag(Lam(:,k));
        D=M(:,:,k)-A*L*transpose(A);
        Cls(Nit)=Cls(Nit)+w(k)*sum(sum(D.*conj(D)));
    end
    
    if Nit>1
        if abs(Cls(Nit)-Cls(Nit-1))<TOL
            break
        end
    end
    
end
Cls=Cls(1:Nit);
