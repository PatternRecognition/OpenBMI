function [tm,T]=loctime(sig);
%LOCTIME Time localization caracteristics.
%	[TM,T]=LOCTIME(SIG) computes the time localization
%	caracteristics of signal SIG. 
% 
%	SIG is the signal.
%	TM  is the averaged time center.
%	T   is the time spreading.
%
%	Example :
%	 z=amgauss(160,80,50); [tm,T]=loctime(z)
%
%	See also LOCFREQ.

%	F. Auger, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

[sigr,sigc]=size(sig);
if (sigc~=1),
 error('The signal must have 1 column');
else
 sig2=abs(sig).^2; sig2=sig2/mean(sig2);
 t=(1:sigr)';
 tm=mean(t.*sig2);
 T=2*sqrt(pi*mean((t-tm).^2 .* sig2)); 
end;
