function noise=noisecu(N);
%NOISECU Analytic complex uniform white noise.
%	NOISE=NOISECU(N) computes an analytic complex white uniform
%	 noise of length N with mean 0.0 and variance 1.0. 
%
%	Examples :
%	 N=512;noise=noisecu(N);mean(noise),std(noise).^2
%	 subplot(211); plot(real(noise)); axis([1 N -1.5 1.5]);
%	 subplot(212); f=linspace(-0.5,0.5,N); 
%	 plot(f,abs(fftshift(fft(noise))).^2);
%		
%	See also RAND, RANDN, NOISECG.

%	O. Lemoine, June 95/May 96 - F. Auger, August 95.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 


if (N <= 0),
 error ('The signal length N must be strictly positive' );
end;

rand('uniform');

if N<=2,
 noise=(rand(N,1)-0.5+j*(rand(N,1)-0.5))*sqrt(6); 
else
 noise=rand(2^nextpow2(N),1)-0.5;
end

if N>2,
 noise=hilbert(noise)/std(noise)/sqrt(2);
 noise=noise(length(noise)-(N-1:-1:0));
end