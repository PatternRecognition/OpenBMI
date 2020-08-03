function [Y, W, V, b, opt]=sobi1(X,n,p)
%% sobi1() - Second Order Blind Identification (SOBI) by joint diagonalization of
%          correlation  matrices. THIS CODE ASSUMES TEMPORALLY CORRELATED SIGNALS,
%          and uses correlations across times in performing the signal separation. 
%          Thus, estimated time delayed covariance matrices must be nonsingular 
%          for at least some time delays. 
% Usage:  
%         >>  [Y, W, D, b, opt]= sobi1(X,n,p)
% Inputs: 
%   data - data matrix of size [N,m] ELSE of size [N,m,t] where
%                N is the  number of samples, 
%                m is the number of sensors,
%                t is the  number of trials (here, correlations avoid epoch boundaries)
%      n - number of sources {Default: n=m}
%      p - number of correlation matrices to be diagonalized {Default: min(100, N/3)}
%          Note that for noisy data, the authors strongly recommend using at least 100 
%          time delays.
%
% Outputs:
% Y - SOBI output signal
% W - matrix containg the SOBI ICA filters in the columns
% D - delta values, measure of slowness
% b - bias vector
% opt
%   M - stacked covariance Matrix
%   winv - Matrix of size [m,n], an estimate of the *mixing* matrix. Its
%          columns are the component scalp maps. NOTE: This is the inverse
%          of the usual ICA unmixing weight matrix. Sphering (pre-whitening),
%          used in the algorithm, is incorporated into winv. i.e.,
%             >> icaweights = pinv(winv); icasphere = eye(m);
%   act  - matrix of dimension [n,N] an estimate of the source activities 
%             >> data            = winv            * act; 
%                [size m,N]        [size m,n]        [size n,N]
%             >> act = pinv(winv) * data;
%
% Authors:  A. Belouchrani and A. Cichocki (papers listed in function source)
%
% Note: Adapted by Arnaud Delorme and Scott Makeig to process data epochs
%
%
% REFERENCES:
% A. Belouchrani, K. Abed-Meraim, J.-F. Cardoso, and E. Moulines, ``Second-order
%  blind separation of temporally correlated sources,'' in Proc. Int. Conf. on
%  Digital Sig. Proc., (Cyprus), pp. 346--351, 1993.
%
%  A. Belouchrani and K. Abed-Meraim, ``Separation aveugle au second ordre de
%  sources correlees,'' in  Proc. Gretsi, (Juan-les-pins), 
%  pp. 309--312, 1993.
%
%  A. Belouchrani, and A. Cichocki, 
%  Robust whitening procedure in blind source separation context, 
%  Electronics Letters, Vol. 36, No. 24, 2000, pp. 2050-2053.
%  
%  A. Cichocki and S. Amari, 
%  Adaptive Blind Signal and Image Processing, Wiley,  2003.
%
%
% taken from http://sccn.ucsd.edu/eeglab/allfunctions/sobi.html and adapted
% to make it usable in the BBCI toolbox
%
% JohannesHoehne 12/2011







% permute dimensions
% [m,N,t] --> [N,m,t]
if length(size(X)) == 2
    X = permute(X, [2 1]);
elseif length(size(X)) == 3
    X = permute(X, [2 1 3]);
else 
    error('X has to be 2- or 3-dimensional!');
end
[m,N,ntrials]=size(X);

if nargin<1 | nargin > 3
  help sobi
elseif nargin==1,
 n=m; % Source detection (hum...)
 p=min(100,ceil(N/3)); % Number of time delayed correlation matrices to be diagonalized 
 % Authors note: For noisy data, use at least p=100 the time-delayed covariance matrices.
elseif nargin==2,
 p=min(100,ceil(N/3)); % Default number of correlation matrices to be diagonalized
                       % Use < 100 delays if necessary for short data epochs
end; 

%
% Make the data zero mean
%
b = mean(X,2);
X = X - repmat(b, 1, size(X,2));

%
% Pre-whiten the data based directly on SVD
%
[UU,Y,VV]=svd(X(:,:)',0);
Q= pinv(Y)*VV';
X(:,:)=Q*X(:,:);

% Alternate whitening code
% Rx=(X*X')/T;
% if m<n, % assumes white noise
%   [U,D]=eig(Rx); 
%   [puiss,k]=sort(diag(D));
%   ibl= sqrt(puiss(n-m+1:n)-mean(puiss(1:n-m)));
%    bl = ones(m,1) ./ ibl ;
%   BL=diag(bl)*U(1:n,k(n-m+1:n))';
%   IBL=U(1:n,k(n-m+1:n))*diag(ibl);
% else    % assumes no noise
%    IBL=sqrtm(Rx);
%    Q=inv(IBL);
% end;
% X=Q*X;

%
% Estimate the correlation matrices
%
 k=1;
 pm=p*m; % for convenience
 for u=1:m:pm, 
   k=k+1; 
   for t = 1:ntrials 
       if t == 1
           Rxp=X(:,k:N,t)*X(:,1:N-k+1,t)'/(N-k+1)/ntrials;
       else
           Rxp=Rxp+X(:,k:N,t)*X(:,1:N-k+1,t)'/(N-k+1)/ntrials;
       end;
   end;
   M(:,u:u+m-1)=norm(Rxp,'fro')*Rxp;  % Frobenius norm =
 end;                                  % sqrt(sum(diag(Rxp'*Rxp)))

%
% Perform joint diagonalization
%
epsil=1/sqrt(N)/100; 
encore=1; 
V=eye(m);
while encore, 
 encore=0;
 for p=1:m-1,
  for q=p+1:m,
   % Perform Givens rotation
   g=[   M(p,p:m:pm)-M(q,q:m:pm)  ;
         M(p,q:m:pm)+M(q,p:m:pm)  ;
      i*(M(q,p:m:pm)-M(p,q:m:pm)) ];
	  [vcp,D] = eig(real(g*g')); 
          [la,K]=sort(diag(D));
   angles=vcp(:,K(3));
   angles=sign(angles(1))*angles;
   c=sqrt(0.5+angles(1)/2);
   sr=0.5*(angles(2)-j*angles(3))/c; 
   sc=conj(sr);
   oui = abs(sr)>epsil ;
   encore=encore | oui ;
   if oui , % Update the M and V matrices 
    colp=M(:,p:m:pm);
    colq=M(:,q:m:pm);
    M(:,p:m:pm)=c*colp+sr*colq;
    M(:,q:m:pm)=c*colq-sc*colp;
    rowp=M(p,:);
    rowq=M(q,:);
    M(p,:)=c*rowp+sc*rowq;
    M(q,:)=c*rowq-sr*rowp;
    temp=V(:,p);
    V(:,p)=c*V(:,p)+sr*V(:,q);
    V(:,q)=c*V(:,q)-sc*temp;
   end%% if
  end%% q loop
 end%% p loop
end%% while

%
% Estimate the mixing matrix 
%
winv = pinv(Q)*V; 

W = pinv(winv)';

%
% Estimate the source activities
%
Y=V'*X(:,:); % estimated source activities

if nargout>4
  opt = [];
  opt.winv = winv;
  opt.M = M;
end
