function [sig,iflaws]=griffitc(N,SNR);
%GRIFFITC Test signal example C of Griffiths' paper. 
%	[SIG,IFLAWS]=GRIFFITC(N,SNR) generates the test signal of  
%	the example C of the paper of Griffiths.
%
%	N      : signal length         (default: 200)
%	SNR    : signal to noise ratio (default: 25 dB)
%	SIG    : output signal
%	IFLAWS : instantaneous frequency laws of the 3 components
%
%	Example :
%	 [sig,iflaws]=griffitc; plotifl(1:200,iflaws); grid;

%	F. Auger, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%	Ref: L.J. Griffiths, "Rapid measurement of digital instantaneous
%	  frequency", IEEE Trans on ASSP, Vol 23, No 2, pp 207-222, 1975.
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin==0),
 N=200; SNR=25;
elseif (nargin==1),
 SNR=25;
end;
[sig1,iflaw1]=fmsin(N,0.25-0.08,0.25+0.08,192.6,50,0.285,+1);
[sig2,iflaw2]=fmsin(N,0.28-0.03,0.28+0.03,110.6,50,0.294,-1);
[sig3,iflaw3]=fmsin(N,0.40-0.02,0.40+0.02,149.6,50,0.417,-1);
noise=hilbert(randn(N,1));
sig=sigmerge(sig1+sig2+sig3,noise,SNR);
if (nargout==2),
 iflaws=[iflaw1 iflaw2 iflaw3];
end;

