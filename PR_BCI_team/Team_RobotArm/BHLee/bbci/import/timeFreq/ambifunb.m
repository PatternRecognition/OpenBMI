function [naf,tau,xi]=ambifunb(x,tau,N,trace);
%AMBIFUNB Narrow-band ambiguity function.
%	[NAF,TAU,XI]=AMBIFUNB(X,TAU,N,TRACE) computes the narrow-band 
%	ambiguity function of a signal X, or the cross-ambiguity 
%	function between two signals.
%	
%	X     : signal if auto-AF, or [X1,X2] if cross-AF (length(X)=Nx).
%	TAU   : vector of lag values     (default : -Nx/2:Nx/2).
%	N     : number of frequency bins (default : length(X)).
%	TRACE : if nonzero,              (default : 0)
%	        the progression of the algorithm is shown.
%	NAF   : doppler-lag representation, with the doppler bins stored 
%	        in the rows and the time-lags stored in the columns.
%	        When called without output arguments, AMBIFUNB displays
%	        the squared modulus of the ambiguity function by means of
%	        contour.
%	XI    : vector of doppler values.
%
%	Example :
%	 sig=anabpsk(256,8);
%	 ambifunb(sig); 
%
%	See also AMBIFUWB.

%	O. Lemoine, F. Auger - August 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin == 0),
 error('At least one parameter required');
end;
[xrow,xcol] = size(x);
if (xcol==0)|(xcol>2),
 error('X must have one or two columns');
end;

if (nargin == 1),
 if rem(xrow,2)==0, tau=(-xrow/2+1):(xrow/2-1); 
 else tau=(-(xrow-1)/2):((xrow+1)/2-1); end
 N=xrow; trace=0;
elseif (nargin == 2),
 N=xrow ; trace=0;
elseif (nargin == 3),
 trace=0;
end;

[taurow,taucol] = size(tau);

if (taurow~=1),
 error('TAU must only have one row'); 
elseif (N<0),
 error('N must be greater than zero');
end;

naf=zeros(N,taucol); 
if trace, disp('Narrow-band ambiguity function'); end;
for icol=1:taucol,
 if trace, disprog(icol,taucol,10); end;
 taui=tau(icol);
 t=(1+abs(taui)):(xrow-abs(taui));
 naf(t,icol)=x(t+taui,1).* conj(x(t-taui,xcol));
end;

naf=fft(naf);
naf=naf([(N+rem(N,2))/2+1:N 1:(N+rem(N,2))/2],:);

xi=(-(N-rem(N,2))/2:(N+rem(N,2))/2-1)/N;

if (nargout==0),
 contour(2*tau,xi,abs(naf).^2,16); 
 grid
 xlabel('Delay'); ylabel('Doppler'); shading interp
 title('Narrow-band ambiguity function');
end;
