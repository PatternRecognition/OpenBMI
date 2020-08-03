function y = amgauss(N,t0,T);
%AMGAUSS Generate gaussian amplitude modulation.
%	Y=AMGAUSS(N,T0,T) generates a gaussian amplitude modulation 
%	centered on a time T0, and with a spread proportional to T.
%	This modulation is scaled such that Y(T0)=1 
%	and Y(T0+T/2) and Y(T0-T/2) are approximately equal to 0.5 .
% 
%	N  : number of points.
%	T0 : time center		(default : N/2).
%	T  : time spreading		(default : 2*sqrt(N)). 
%	Y  : signal.
%
%	Examples:
%	 z=amgauss(160); plot(z);
%	 z=amgauss(160,90,40); plot(z);
%	 z=amgauss(160,180,50); plot(z);
%
%	See also AMEXPO1S, AMEXPO2S, AMRECT, AMTRIANG.

%	F. Auger, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin == 0),
 error ( 'The number of parameters must be at least 1.' );
elseif (nargin == 1),
 t0=N/2; T=2*sqrt(N);
elseif (nargin ==2),
 T=2*sqrt(N);
end;

if (N<=0),
 error('N must be greater or equal to 1.');
else
 tmt0=(1:N)'-t0;
 y = exp(-(tmt0/T).^2 * pi);
end;
