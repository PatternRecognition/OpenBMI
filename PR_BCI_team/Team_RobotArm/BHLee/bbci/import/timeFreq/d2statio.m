function [d,f]=d2statio(sig);
%D2STATIO Distance to stationarity
%	[D,F]=D2STATIO(SIG) evaluates the distance of the signal
%	to stationarity, using the pseudo Wigner-Ville distribution.
%
%	SIG : signal to be analyzed (real or complex).
%	D   : vector giving the distance to stationarity for each frequency.
%	F   : vector of frequency bins
%
%	Example :
%	 sig=noisecg(128); [d,f]=d2statio(sig); plot(f,d);
%	 xlabel('Frequency'); ylabel('Distance'); 
%
%	 sig=fmconst(128); [d,f]=d2statio(sig); plot(f,d);
%	 xlabel('Frequency'); ylabel('Distance'); 
%
 
%	O. Lemoine - May 1996.
%	Copyright (c) by CNRS France, 1996.
%       send bugs to f.auger@ieee.org

N=length(sig);

[tfr,t,f]=tfrspwv(sig);
 
d2=((tfr-mean(tfr')'*ones(1,N))/norm(sig)).^2;	

d=mean(d2')';

