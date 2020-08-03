function y=odd(x);
%ODD	Round towards nearest odd value.
%	Y=ODD(X) rounds each element of X towards the nearest odd
%	integer value. If an element of X is even, ODD adds +1 to 
% 	this value. X can be a scalar, a vector or a matrix.
%
%	X : scalar, vector or matrix to be rounded
%	Y : output scalar, vector or matrix containing only odd values
%
%	Example :
%	 X=[1.3 2.08 -3.4 90.43]; Y=odd(X)
% 
%	See also ROUND, CEIL, FIX, FLOOR.

%	O. Lemoine - August 1996.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin == 0),
 error('There must be an input parameter');
end;

[xrow,xcol]=size(x);
y=zeros(xrow,xcol);

for k=1:xrow,
 for l=1:xcol,  
  y(k,l)=floor(x(k,l));
  if rem(y(k,l),2)==0,
   y(k,l)=ceil(x(k,l));
  end
  if rem(y(k,l),2)==0,
   y(k,l)=y(k,l)+1;
  end
 end
end