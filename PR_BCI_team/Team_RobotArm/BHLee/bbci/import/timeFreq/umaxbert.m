function y=umaxbert(u);
%UMAXBERT Determination of the maximum value of u for Bertrand distribution.
%	Y=UMAXBERT(u) is the function Y(u)=(H(u)+u/2)/(H(u)-u/2)-fmax/fmin. 
%	Doing UMAX = fzero('umaxbert',0); gives the maximum value for U in the
%	computation of the Bertrand distribution. For this distribution, 
%	 	 H(u) = (u/2)*coth(u/2).
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

y = exp(u)-ratio_f;