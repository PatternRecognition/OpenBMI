function sig=izak(dzt)
%IZAK 	Inverse Zak transform.
%	SIG=IZAK(DZT) computes the inverse Zak transform of matrix DZT.
%
%	DZT : (N,M) matrix of Zak samples.
%	SIG : Output signal (M*N,1) containing the inverse Zak transform.
%
%	Example :
%	 sig=fmlin(256); DZT=zak(sig); sigr=izak(DZT);
%	 plot(real(sigr)); hold; plot(real(sig)); hold;
%
%	See also ZAK, TFRGABOR.

%	O. Lemoine - February 1996
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

[N,M]=size(dzt);

if nargin<1,
  error('The number of parameters must be one');
end

sig=zeros(N*M,1);

for m=1:M,
  sig(m+(0:N-1)*M)=sqrt(N)*ifft(dzt(:,m));
end

