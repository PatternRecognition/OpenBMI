function [x,iflaw] = fmhyp(N,p1,p2)
%FMHYP	Signal with hyperbolic frequency modulation.
%	[X,IFLAW]=FMHYP(N,P1,P2) generates a signal with hyperbolic
%	frequency modulation : X(t) = exp(i.2.pi(F0.t + C/log|t|)) 
%
%	N  : number of points in time
%	P1 : if the number of input arguments (NARGIN) is 2, P1 is a 
%	    vector containing the two coefficients [F0 C] for an
%	    hyperbolic instantaneous frequency or [T0 C] for an
%	    hyperbolic group delay (sampling frequency is set to 1).
%	    If NARGIN=3, P1 (as P2) is a time-frequency point of the 
%	    form [Ti Fi]. Ti is in seconds and Fi is a normalized frequency
%	    (between 0 and 0.5). The coefficients F0 and C are then deduced
%	    such that the frequency modulation law fits the points P1 and P2.
%	P2 : same as P1 is NARGIN=3         (optional)
%	X  : time row vector containing the modulated signal samples 
%	IFLAW : instantaneous frequency law
%
%	Examples :   
%	 [X,IFLAW]=fmhyp(128,[1 .5],[32 0.1]);
%	 subplot(211); plot(real(X));
%	 subplot(212); plot(IFLAW);
%
%	See also FMLIN, FMSIN, FMPAR, FMCONST, FMODANY, FMPOWER.

%	P. Goncalves - October 1995, O. Lemoine - November 1995
%	Copyright (c) 1995 Rice University
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin <= 1),
  error ( 'The number of parameters must be at least 3.' );
elseif (N <= 0),
 error ('The signal length N must be strictly positive' );
elseif (nargin == 2)
  if length(p1) ~= 2
    error('Bad number of coefficients for P1');
  end
  f0 = p1(1) ;
  c  = p1(2) ;
elseif (nargin == 3) ;
  if (length(p1) ~= 2) |(length(p2) ~= 2),
    error('Bad number of coefficients for P1 or P2');
  end
  if p1(1)>N | p1(1)<1,
   error ('P1(1) must be between 1 and N');
  elseif p2(1)>N | p2(1)<1,
   error ('P2(1) must be between 1 and N');
  elseif p1(2)<0,
   error ('P1(2) must be > 0');
  elseif p2(2)<0,
   error ('P2(2) must be > 0');
  end
  c = (p2(2) - p1(2))/(1/p2(1) - 1/p1(1)) ;
  f0 = p1(2) - c/p1(1) ;
end  

t = 1:N ;

phi = 2*pi*(f0*t + c*log(abs(t))); 
iflaw = (f0 + c*abs(t).^(-1)).' ;

aliasing = find(iflaw < 0 | iflaw > 0.5) ;
if isempty(aliasing) == 0
 disp(['!!! WARNING: signal is undersampled or has negative frequencies']) ;
end

x = exp(i*phi).';
