function y=anastep(N,ti);
%ANASTEP Analytic projection of unit step signal.
%	Y=ANASTEP(N,TI) generates the analytic projection of a
%	unit step signal.
%	
%	N  : number of points.
%	TI : starting position of the unit step.
%	
%	Examples :
%	 signal=anastep(256,128);plot(real(signal));
%	 signal=-2.5*anastep(512,301);plot(real(signal));
%
%	See also ANASING, ANAFSK, ANABPSK, ANAQPSK, ANAASK.

%	O. Lemoine - June 1995, F. Auger, August 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%			    lemoine@alto.unice.fr 

if (nargin==0),
 error('at least one parameter required');
elseif (nargin==1),
 ti=round(N/2);
end;

if (N<=0),
 error('N must be greater than zero');
else
 t=(1:N)';
 y=hilbert(t>=ti);
end;
