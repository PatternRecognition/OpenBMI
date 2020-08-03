function H=kaytth(length);
%KAYTTH	 Kay-Tretter filter computation. 
%	H=KAYTTH(length); Kay-Tretter filter computation.
% 
%	See also INSTFREQ

%	F. Auger, March 1994.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

pp1=length*(length+1);
den=2.0*length*(length+1)*(2.0*length+1.0)/3.0;
i=1:length; H=pp1-i.*(i-1);

H=H ./ den;

