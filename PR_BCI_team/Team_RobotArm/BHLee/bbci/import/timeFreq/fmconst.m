function [y,iflaw] = fmconst(N,fnorm,t0);
%FMCONST Signal with constant frequency modulation.
%	[Y,IFLAW] = FMCONST(N,FNORM,T0) generates a frequency modulation  
%	with a constant frequency fnorm.
%	The phase of this modulation is such that y(t0)=1.
% 
%	N     : number of points.
%	FNORM : normalised frequency.       (default: 0.25)
%	T0    : time center.                (default: N/2 )
%	Y     : signal.
%	IFLAW : instantaneous frequency law (optional).
%
%	Example: 
%	 z=amgauss(128,50,30).*fmconst(128,0.05,50); plot(real(z));
%
%	See also FMLIN, FMSIN, FMODANY, FMHYP, FMPAR, FMPOWER.

%	F. Auger, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%			    lemoine@alto.unice.fr 

if (nargin == 0),
 error ( 'The number of parameters must be at least 1.' );
elseif (nargin == 1),
 t0=N/2; fnorm=0.25 ;
elseif (nargin ==2),
 t0=N/2;
end;

if (N<=0),
 error('N must be greater or equal to 1.');
elseif (abs(fnorm)>0.5),
 error('The normalised frequency must be between -0.5 and 0.5');
else
 tmt0=(1:N)'-t0;
 y = exp(j*2.0*pi*fnorm*tmt0);
 y=y/y(t0);
 if (nargout==2), iflaw=fnorm*ones(N,1); end;
end;

