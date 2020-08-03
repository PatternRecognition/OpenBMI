function [y,iflaw]=fmodany(iflaw,t0);
%FMODANY Signal with arbitrary frequency modulation.
%	[Y,IFLAW]=FMODANY(IFLAW,T0) generates a frequency modulated
%	signal whose instantaneous frequency law is approximately given by
%	the vector IFLAW (the integral is approximated by CUMSUM).
%	The phase of this modulation is such that y(t0)=1.
% 
%	IFLAW : vector of the instantaneous frequency law samples.
%	T0    : time reference		(default: 1).
%	Y     : output signal
%
%	Example:
%	 [y1,ifl1]=fmlin(100); [y2,ifl2]=fmsin(100);
%	 iflaw=[ifl1;ifl2]; sig=fmodany(iflaw); 
%	 subplot(211); plot(real(sig))
%	 subplot(212); plot(iflaw); 
%
%	See also FMCONST, FMLIN, FMSIN, FMPAR, FMHYP, FMPOWER.

%	F. Auger, August 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

[ifrow,ifcol]=size(iflaw);
if (ifcol~=1),
 error('IFLAW must have one column');
elseif (max(abs(iflaw))>0.5),
 error('The elements of IFLAW should not be higher than 0.5');
end;
if (nargin==1),
 t0=1;
elseif (t0==0)|(t0>ifrow),
 error('T0 should be between 1 and length(IFLAW)');
end;

y=exp(j*2.0*pi*cumsum(iflaw));
y=y*conj(y(t0));
