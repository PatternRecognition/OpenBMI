function [fm,B]=locfreq(sig);
%LOCFREQ Frequency localization caracteristics.
%	[FM,B]=LOCFREQ(SIG) computes the frequency localization 
%	caracteristics of signal SIG.
% 
%	SIG   is the signal.
%	FM    is the averaged normalized frequency center.
%	B     is the frequency spreading.
%
%	Example :
%	 z=amgauss(160,80,50);[tm,T]=loctime(z),[fm,B]=locfreq(z),B*T
%
%	See also LOCTIME.

%	F. Auger, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

[N,sigc]=size(sig);
if (sigc~=1),
 error('The signal must have 1 column');
else
 No2r=round(N/2);
 No2f=fix(N/2);
 Sig=fft(sig);
 Sig2=abs(Sig).^2;
 Sig2=Sig2/mean(Sig2);
 freqs=[0:No2f-1 -No2r:-1]'/N;
 fm=mean(freqs.*Sig2);
 B=2*sqrt(pi*mean((freqs-fm).^2.*Sig2));
end;
