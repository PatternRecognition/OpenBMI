function y=modulo(x,N);
%MODULO	Congruence of a vector.
%	Y=MODULO(X,N) gives the congruence of each element of the
%	vector X modulo N. These values are strictly positive and 
%	lower equal than N.
%
%	See also REM.

%	O. Lemoine - February 1996.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

L=length(x);

for l=1:L,
  if rem(x(l),N)==0,
    y(l)=N;
  elseif rem(x(l),N)<0,
    y(l)=rem(x(l),N)+N;
  else
    y(l)=rem(x(l),N);
  end
end
