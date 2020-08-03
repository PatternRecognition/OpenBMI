function [y,am]=anabpsk(N,Ncomp,f0);
%ANABPSK Binary Phase Shift Keying (BPSK) signal.
% 	[Y,AM]=ANABPSK(N,NCOMP,F0) returns a succession of complex
%	sinusoids of NCOMP points each, with a normalized frequency F0 and
%	an amplitude equal to -1 or +1, according to a discrete	uniform
%	law. Such signal is only 'quasi'-analytic.
%	
%	N     : number of points
%	NCOMP : number of points of each component (default: N/5)
% 	F0    : normalized frequency.              (default: 0.25)
% 	Y     : signal
% 	AM    : resulting amplitude modulation     (optional).
%
%	Example :
%	 [signal,am]=anabpsk(300,30,0.1); clg
%  	 subplot(211); plot(real(signal)); subplot(212); plot(am);
%
%	See also ANAFSK, ANAQPSK, ANAASK.

%	O. Lemoine - June 1995, F. Auger - August 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin == 0),
 error('The number of parameters must be at least 1.');
elseif (nargin == 1),
 Ncomp=round(N/5); f0=0.25;
elseif (nargin == 2),
 f0=0.25;
end;

if (N <= 0),
 error('The signal length N must be strictly positive' );
elseif (f0<0)|(f0>0.5),
 error('f0 must be between 0 and 0.5');
end;

rand('uniform'); m=ceil(N/Ncomp);
jumps=2.0*round(rand(m,1))-1.0;
am=kron(jumps,ones(Ncomp,1)); am=am(1:N,1);
y=am.*fmconst(N,f0,1);
