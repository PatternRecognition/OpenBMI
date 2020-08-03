function [fv, P, b]= proc_pr_jade(fv, retain, varargin)
%Cardoso's JADE (ICA)
%fv.x --> fv.x*P - b
%
%[fv, P, b]= proc_pr_jade(fv, retain, <policy>)
%[fv, P, b]= proc_pr_jade(fv, retain, <opt>)
%
% IN  fv     - struct of feature vectors. fv.x is a matrix [T n] for n
%              sensors, T timesteps
%     retain - threshold for determining how many features to retain,
%              depends on opt.policy
%     opt    propertylist or struct of options:
%      .policy - one of 'number_of_features', 'perc_of_features',
%                'perc_of_score': determines the strategy how to choose
%                the number of features to be selected
%
% OUT  fv    - struct of new (poss. reduced) feature vectors. fv.x 
%              is a [T retain] matrix
%      P     - projection matrix
%      b     - mean value before centering
%
%
% In a first step the data is centered and projected to the most
% powerful principal components using the specified 'policy', and
% whitened. The remaining rotation is achieved by a simultaneous
% diagonalisation of 'slices' of the 4th order cumulant tensor.
%
% based on original 'jadeR.m' function by J.-F. Cardoso 
% (see http://sig.enst.fr/~cardoso/stuff.html)

% fcm 16jul2004 
% fcm 08apr2005: changed all fv.x to fv.x' to meet the toolbox standard
% Anton Schwaighofer, Sep 2005
% $Id: proc_pr_jade.m,v 1.3 2005/09/19 15:25:13 neuro_toolbox Exp $


% the default settings
defopt.policy = 'number_of_features';
opt.whitening = 1;

% read the parameters
if ~exist('retain','var')|isempty(retain),
  opt.policy = 'perc_of_features';
  retain = 100;
end;
if length(varargin)==1,
  if isstruct(varargin{1}),
    opt = varargin{1};
  else
    error('Optional parameters should always go by pairs or as fields of a struct');
  end;
elseif length(varargin)>1,
  if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs or as fields of a struct');
  else
    opt = propertylist2struct(varargin{:});
  end;
end;
if ~exist('opt','var'),
  opt = defopt;
else
  opt = set_defaults(opt, defopt);
end;


% perform PCA for dimensionality reduction, whitening. PCA expects the
% data the other way around: [dim Nsamples]
[X, PCAdata] = proc_pr_pca(fv.x', retain, opt);

% Cardosos code expects data in the format [n T] for n sensors, T timesteps
[n T] = size(X);
% Number of components to retain, these are all PCA components
m=n;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code copied from Cardoso's jadeR.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% Estimation of the cumulant matrices.
%   ====================================

dimsymm 	= (n*(n+1))/2;	% Dim. of the space of real symm matrices
nbcm 		= dimsymm  ; 	% number of cumulant matrices
CM 		= zeros(n,n*nbcm);  % Storage for cumulant matrices
R 		= eye(n);  	%% 
Qij 		= zeros(n);	% Temp for a cum. matrix
Xim		= zeros(1,n);	% Temp
Xjm		= zeros(1,n);	% Temp
scale		= ones(n,1)/T ; % for convenience



%% I am using a symmetry trick to save storage.  I should write a
%% short note one of these days explaining what is going on here.
%%

Range = 1:m ; % will index the columns of CM where to store the cum. mats.
for im = 1:m
  Xim = X(im,:) ;
  Qij = ((scale* (Xim.*Xim)) .* X ) * X' - R - 2 * R(:,im)*R(:,im)' ;
  CM(:,Range)= Qij ; 
  Range = Range + m ; 
  for jm = 1:im-1
    Xjm = X(jm,:) ;
    Qij = ( (scale*(Xim.*Xjm)).*X ) *X' - R(:,im)*R(:,jm)' - R(:,jm)*R(:,im)';
    CM(:,Range)	= sqrt(2)*Qij ;  
    Range 		= Range + m ;
  end ;
end;



%%% joint diagonalization of the cumulant matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Init
if 1, 	
  %% Init by diagonalizing a *single* cumulant matrix.  It seems to save
  %% some computation time `sometimes'.  Not clear if initialization is
  %% a good idea since Jacobi rotations are very efficient.
  
  [V,D]	= eig(CM(:,1:m)); % For instance, this one
  for u=1:m:m*nbcm,	 % updating accordingly the cumulant set given the init
    CM(:,u:u+m-1) = CM(:,u:u+m-1)*V ; 
  end;
  CM = V'*CM;
else,	%% The dont-try-to-be-smart init
  V = eye(m) ; % la rotation initiale
end;

seuil	= 1/sqrt(T)/100; % A statistically significant threshold
encore	= 1;
sweep	= 0;
updates = 0;
g	= zeros(2,nbcm);
gg	= zeros(2,2);
G	= zeros(2,2);
c	= 0 ;
s 	= 0 ;
ton	= 0 ;
toff	= 0 ;
theta	= 0 ;

%% Joint diagonalization proper
while encore, 
  encore=0;   
  sweep=sweep+1;
  for p=1:m-1,
    for q=p+1:m,
      Ip = p:m:m*nbcm ;
      Iq = q:m:m*nbcm ;
      
      %%% computation of Givens angle
      g	= [ CM(p,Ip)-CM(q,Iq) ; CM(p,Iq)+CM(q,Ip) ];
      gg	= g*g';
      ton 	= gg(1,1)-gg(2,2); 
      toff 	= gg(1,2)+gg(2,1);
      theta	= 0.5*atan2( toff , ton+sqrt(ton*ton+toff*toff) );
      
      %%% Givens update
      if abs(theta) > seuil,	encore = 1 ;
	updates = updates + 1;
	c	= cos(theta); 
	s	= sin(theta);
	G	= [ c -s ; s c ] ;
	
	pair 		= [p;q] ;
	V(:,pair) 	= V(:,pair)*G ;
	CM(pair,:)	= G' * CM(pair,:) ;
	CM(:,[Ip Iq]) 	= [ c*CM(:,Ip)+s*CM(:,Iq) -s*CM(:,Ip)+c*CM(:,Iq) ] ;
	
	%% fprintf('jade -> %3d %3d %12.8f\n',p,q,s);
	
      end%%of the if
    end%%of the loop on q
  end%%of the loop on p
end%%of the while loop


%%% A separating matrix
%   ===================
B	= V'*PCAdata.P;
%%% We permut its rows to get the most energetic components first.
%%% Here the **signals** are normalized to unit variance.  Therefore,
%%% the sort is according to the norm of the columns of A = pinv(B)

A		= pinv(B);%iW*V ;
[vars,keys]	= sort(sum(A.*A)) ;
B		= B(keys,:);
B		= B(m:-1:1,:) ; % Is this smart ?

% Signs are fixed by forcing the first column of B to have
% non-negative entries.
b	= B(:,1) ;
signs	= sign(sign(b)+0.1) ; % just a trick to deal with sign=0
B	= diag(signs)*B ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of copied code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P = B';
b = (V'*PCAdata.b)';
fv.x = (V'*X)';

warning('Anton: There is still a bug in this routine');
