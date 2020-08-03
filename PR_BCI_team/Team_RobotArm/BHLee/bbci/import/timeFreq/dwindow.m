function Dh=dwindow(h);
%DWINDOW Derive a window.
%	DH=DWINDOW(H) derives a window H.
%
%	Example : 
%	 plot(dwindow(window(210,'hanning')))
%
%	See also WINDOW.

%	F. Auger, August 1994, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin==0),
 error('one parameter required'); 
end;
[hrow,hcol]=size(h); 
if (hcol~=1),
 error('h must have only one column');
end;

Lh=(hrow-1)/2;
step_height=(h(1)+h(hrow))/2;
ramp=(h(hrow)-h(1))/(hrow-1);
h2=[0;h-step_height-ramp*(-Lh:Lh).';0]; 
Dh=(h2(3:hrow+2)-h2(1:hrow))/2 + ramp; 
Dh(1)   =Dh(1)   +step_height; 
Dh(hrow)=Dh(hrow)-step_height;

