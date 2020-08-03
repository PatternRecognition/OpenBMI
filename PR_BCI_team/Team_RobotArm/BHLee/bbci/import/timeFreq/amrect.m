function y = amrect(N,t0,T);
%AMRECT	Generate rectangular amplitude modulation.
%	Y = AMRECT(N,T0,T) generates a rectangular amplitude modulation 
%	centered on a time T0, and with a spread proportional to T.
%	This modulation is scaled such that Y(T0)=1.
% 
%	N  : number of points.
%	T0 : time center		(default : N/2).
%	T  : time spreading		(default : 2*sqrt(N)).
%	Y  : signal.
%
%	Examples:
%	 z=amrect(160);plot(z);
%	 z=amrect(160,90,40);plot(z);
%	 z=amrect(160,180,70);plot(z);
%
%	See also AMEXPO1S, AMEXPO2S, AMGAUSS, AMTRIANG.

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
 y = (abs(tmt0) <= 0.5*T*sqrt(3.0/pi));
end;
