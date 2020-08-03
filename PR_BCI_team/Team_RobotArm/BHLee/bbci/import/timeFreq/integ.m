function som = integ(y,x) 
%INTEG	Approximate integral.
%	SOM=INTEG(Y,X) approximates the integral of vector Y
%	according to X.
%
%	Y   : N-row-vector (or MxN-matrix) to be integrated 
%	      (along each row).  
%	X   : N-row-vector containing the integration path of Y
%				(default : 1:N)
%	SOM : value (or Mx1 vector) of the integral
%
%	Example :    
%	 Y = altes(256,0.1,0.45,10000)'; X = (0:255);
%	 SOM = integ(Y,X)
%
%	See also INTEG2D.

%	P. Goncalves, October 95
%	Copyright (c) 1995 Rice University
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

[M,N]=size(y);

if nargin<1,
 error('At least one parameter required');
elseif nargin==1,
 x=1:N;
end

[Mx,Nx]=size(x);
if (Mx~=1),
 error('X must be a row-vector');
elseif (N~=Nx),
 error('Y must have as many columns as X');
elseif (N==1 & M>1),
 error('Y must be a row-vector or a matrix');
end
 
dy = y(:,1:N-1) + y(:,2:N) ;
dx = (x(2:N)-x(1:N-1))/2 ;
som = dy*dx';
