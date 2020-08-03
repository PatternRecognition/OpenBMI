function y=umaxunte(u);
%UMAXUNTE Determination of the maximum value of u for Unterberger distribution.
%	Y=UMAXUNTE(u) is the function Y(u)=(H(u)+u/2)/(H(u)-u/2)-fmax/fmin. 
%	Doing UMAX = fzero('umaxunte',0); gives the maximum value for U in the
%	computation of the Unterberger distribution. For this distribution, 
%	 	 	H(u) = sqrt(1+(u/2)^2).
%
%	U     : real vector
%	Y     : value of the function (H(u)+u/2)/(H(u)-u/2)-fmax/fmin.

%	P. Goncalves, October 95 - O. Lemoine, July 1996. 
%	Copyright (c) 1995 Rice University - CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

global ratio_f

y = (sqrt(1+(u/2)^2)+u/2)/(sqrt(1+(u/2)^2)-u/2)-ratio_f;