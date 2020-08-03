function y=umaxdfla(u);
%UMAXDFLA Determination of the maximum value of u for D-Flandrin distribution.
%	Y=UMAXDFLA(u) is the function Y(u)=(H(u)+u/2)/(H(u)-u/2)-fmax/fmin. 
%	Doing UMAX = fzero('umaxdfla',0); gives the maximum value for U in the
%	computation of the D-Flandrin distribution. For this distribution, 
%	 	 	H(u) = 1+(u/4)^2.
%
%	U     : real vector
%	Y     : value of the function (H(u)+u/2)/(H(u)-u/2)-FMAX/FMIN.

%	P. Goncalves, October 95 - O. Lemoine, July 1996. 
%	Copyright (c) 1995 Rice University - CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

global ratio_f

y = ((1+u/4)/(1-u/4))^2-ratio_f;