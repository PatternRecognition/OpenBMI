function dlr=ambnb(x,tau,N);
%AMBNB  Narrow-band ambiguity function.
%	DLR=AMBNB(X,TAU,N) computes the narrow-band ambiguity function of 
%       a signal X, or the cross-ambiguity function between two signals.
%	
%	X     : signal if auto-AF, or [X1,X2] if cross-AF.
%	TAU   : vector of lag values.
%	N     : number of frequency bins (default : length(X)).
%	DLR   : doppler-lag representation.

%	O. Lemoine, F. Auger - August 1995.
%	Modified by P. Goncalves - January 1996.
%	(ambiguity matrix reshaped)

if (nargin == 0),
 error('At least one parameter required');
end;
[xrow,xcol] = size(x);
if (nargin == 1),
 if rem(xrow,2)==0, tau=(-xrow/2):(xrow/2-1); 
 else tau=(-(xrow-1)/2):((xrow+1)/2-1); end
 N=xrow ;
elseif (nargin == 2),
 N=xrow ;
end;

[taurow,taucol] = size(tau) ;

if (xcol==0)|(xcol>2),
 error('X must have one or two columns');
elseif (taurow~=1),
 error('TAU must only have one row'); 
elseif (taucol>xrow),
 error('TAU must have values between -length(X)/2+1 and length(X)/2-1');
elseif (N<0),
 error('N must be greater than zero');
end;

dlr=zeros(N,taucol); 
for icol=1:taucol,
 taui=tau(icol);
 t=max(1+taui,1-taui):min([xrow-taui,xrow+taui,N]);
 dlr(t,icol)=x(t+taui,1).*conj(x(t-taui,xcol));
end;
dlr=fft(dlr);

% dlr matrix reshaped (vertical fftshift) - Paulo

ncol = size(dlr,2) ;
for nx = 1 : ncol
  dlr(:,nx) = fftshift(dlr(:,nx)) ;
end
dlr = dlr.' ;
