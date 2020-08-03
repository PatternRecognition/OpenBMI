function [ V , D ] =  utils_proc_pr_tdsep_jdiag(A,jthresh)
% function [ V , D ] =  joint_diag(A,jthresh)
%
% Joint approximate of n (complex) matrices of size m*m stored in the
% m*mn matrix A by minimization of a joint diagonality criterion
%
% Input :
% * the m*nm matrix A is the concatenation of n matrices with size m
%   by m. We denote A = [ A1 A2 .... An ]
% * threshold is an optional small number (typically = 1.0e-8 see below).
%
% Output :
% * V is an m*m unitary matrix.
% * D = V'*A1*V , ... , V'*An*V has the same size as A and is a
%   collection of diagonal matrices if A1, ..., An are exactly jointly
%   unitarily diagonalizable.
%
% ----------------------------------------------------------------
%
% The algorithm finds a unitary matrix V such that the matrices
% V'*A1*V , ... , V'*An*V are as diagonal as possible, providing a
% kind of `average eigen-structure' shared by the matrices A1 ,...,An.
% If the matrices A1,...,An do have an exact common eigen-structure ie
% a common orthonormal set eigenvectors, then the algorithm finds it.
% The eigenvectors THEN are the column vectors of V and D1, ...,Dn are
% diagonal matrices.
% 
% The algorithm implements a properly extended Jacobi algorithm.  The
% algorithm stops when all the Givens rotations in a sweep have sines
% smaller than 'threshold'.
%
% In many applications, the notion of approximate joint
% diagonalization is ad hoc and very small values of threshold do not
% make sense because the diagonality criterion itself is ad hoc.
% Hence, it is often not necessary in applications to push the
% accuracy of the rotation matrix V to the machine precision.
%
% PS: If a numrical analyst knows `the right way' to determine jthresh
%     in terms of 1) machine precision and 2) size of the problem,
%     I will be glad to hear about it.
% 
%
% This version of the code is for complex matrices, but it also works
% with real matrices.  However, simpler implementations are possible
% in the real case.
%
%
%----------------------------------------------------------------
% References:
%
% The 1st paper below presents the Jacobi trick.
% The second paper is a tech. report the first order perturbation
% of joint diagonalizers
%
%
%@article{SC-siam,
%  HTML        = "ftp://sig.enst.fr/pub/jfc/Papers/siam_note.ps.gz",
%  author       = "Jean-Fran\c{c}ois Cardoso and Antoine Souloumiac",
%  journal      = "{SIAM} J. Mat. Anal. Appl.",
%  title        = "Jacobi angles for simultaneous diagonalization",
%  pages        = "161--164",
%  volume       = "17",
%  number       = "1",
%  month        = jan,
%  year         = {1996}
%  }
%
%
%
%@techreport{PertDJ,
%  author       = "Jean-Fran\c{c}ois Cardoso",
%  HTML         = "ftp://sig.enst.fr/pub/jfc/Papers/joint_diag_pert_an.ps",
%  institution  = "T\'{e}l\'{e}com {P}aris",
%  title        = "Perturbation of joint diagonalizers. Ref\# 94D027",
%  year         = "1994"
%}
%
%
%----------------------------------------------------------------
% Author : Jean-Francois Cardoso. cardoso@sig.enst.fr
% Comments, bug reports, etc are welcome.
%
% This software is for non commercial use only.
% It is freeware but not in the public domain.
%----------------------------------------------------------------


[m,nm] = size(A);

V       =eye(m);
g       =zeros(3,m);
G       =zeros(3);

encore=1; compteur=0;

while encore, encore=0;
 for p=1:m-1,
  for q=p+1:m,

   %%% The quadratic form
   g=[   A(p,p:m:nm)-A(q,q:m:nm)  ;
         A(p,q:m:nm)+A(q,p:m:nm)  ;
      i*(A(q,p:m:nm)-A(p,q:m:nm)) ];
   G = real(g*g');

   %%% The Givens parameters in closed form
   [vcp,D] = eig(G);
   [la,K]=sort(diag(D));
   angles=vcp(:,K(3));
   angles=sign(angles(1))*angles;

   c=sqrt(0.5+angles(1)/2);
   sr=0.5*(angles(2)-j*angles(3))/c;
   sc=conj(sr);

   %% Givens update 
   oui = abs(sr)>jthresh ;
   encore=encore | oui ;
   if oui , %%%update of the A and V matrices 
    compteur=compteur+1;
    colp=A(:,p:m:nm);
    colq=A(:,q:m:nm);
      A(:,p:m:nm)=c*colp+sr*colq;
      A(:,q:m:nm)=c*colq-sc*colp;
    rowp=A(p,:);
    rowq=A(q,:);
      A(p,:)=c*rowp+sc*rowq;
      A(q,:)=c*rowq-sr*rowp;
    temp=V(:,p);
    V(:,p)=c*V(:,p)+sr*V(:,q);
    V(:,q)=c*V(:,q)-sc*temp;

   end%% if
  end%% q loop
 end%% p loop
end%% while


D = A ;
