function x=altes(N,fmin,fmax,alpha) ;
%ALTES	Altes signal in time domain.
%	X=ALTES(N,FMIN,FMAX,ALPHA) generates the Altes signal in
%	the time domain.
%
%	N     : number of points in time   
%	FMIN  : lower frequency bound (value of the hyperbolic
%	        instantaneous frequency law at the sample N), 
%	        in normalized frequency             (default : .05) 
%	FMAX  : upper frequency bound (value of the hyperbolic
%	        instantaneous frequency law at the first sample), 
%	        in normalized frequency             (default : 0.5)
%	ALPHA : attenuation factor of the envelope (default : 300)
%	X     : time row vector containing the Altes signal samples.
%
%	Example: 
%	 x=altes(128,0.1,0.45); plot(x);
%
%	See also KLAUDER, ANASING, ANAPULSE, ANASTEP, DOPPLER.

%	P. Goncalves - September 1995.
%	Copyright (c) 1995 Rice University
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin == 0),
 error ( 'The number of parameters must be at least 1.' );
elseif (nargin == 1),
 fmin=0.05; fmax=0.5; alpha=300;
elseif (nargin == 2),
 fmax=0.5; alpha=300;
elseif (nargin == 3),
 alpha=300 ;
end;

if (N <= 0),
 error ('The signal length N must be strictly positive' );
elseif (fmin > 0.5) | (fmin < 0),
 error ( 'FMIN must be in ]0 , 0.5]' ) ;
elseif (fmax > 0.5) | (fmax < 0),
 error ( 'FMAX must be between 0 and 0.5' ) ;
elseif (alpha <= 1),
 error ( 'ALPHA must be > 1' ) ;
else
 g = exp((log(fmax/fmin))^2/(8*log(alpha))) ;
 nu0 = sqrt(fmin*fmax) ;
 beta=sqrt(2*log(g)*log(alpha));
 t0 = N/(exp(beta)-exp(-beta)) ;
 t1 = t0*exp(-beta); t2 = t0*exp(beta) ;
 b = -t0*nu0*g*log(g) ;
 t = linspace(t1,t2,N+1) ; t = t(1:N) ;
 x = (exp(-(log(t./t0).^2)/(2*log(g)))).*cos(2*pi*b*log(t./t0)/log(g)) ;
 x = x.'/norm(x) ;
end

