function y=anapulse(N,ti);
%ANAPULSE Analytic projection of unit amplitude impulse signal.
%	y=ANAPULSE(N,TI) returns an analytic N-dimensional signal 
%	whose real part is a Dirac impulse at t=TI.
%
% 	N  : number of points.
%	TI : time position of the impulse	(default : round(N/2)). 
%	
%	Example :
%	 signal=2.5*anapulse(512,301);plot(real(signal));
%
%	See also ANASTEP, ANASING, ANABPSK, ANAFSK.

%	O. Lemoine - June 1995, F. Auger, August 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin==0),
 error('at least one parameter required');
elseif (nargin==1),
 ti=round(N/2);
end;

if (N<=0),
 error('N must be greater than zero');
else
 t=(1:N)';
 y=hilbert(t==ti);
end;
