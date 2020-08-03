%TFDEMO2 Non stationary signals

%	O. Lemoine - May 1996. 
%	Copyright (c) CNRS.

clc; zoom on; clf; 
echo on;

% Time and frequency localizations and the Heisenberg-Gabor inequality 
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% The time and frequency localizations can be evaluated thanks to 
% the M-files loctime.m and locfreq.m of the Toolbox. The first one
% gives the average time center (tm) and the duration (T) of a signal,
% and the second one the average normalized frequency (num) and the 
% normalized bandwidth (B). For example, for a linear chirp with a 
% Gaussian amplitude modulation, we obtain :

sig=fmlin(256).*amgauss(256); 
subplot(211); plot(real(sig)); axis([1 256 -1 1]); grid;
xlabel('Time'); ylabel('Real part'); title('Signal in time');
dsp=fftshift(abs(fft(sig)).^2);
subplot(212); plot((-128:127)/256,dsp); grid;
xlabel('Normalized frequency'); ylabel('Squared modulus'); 
title('Energy spectrum');
[tm ,T]=loctime(sig) 
[num,B]=locfreq(sig)

% Press any key to continue...
 
pause; clc;

% One interesting property of this product T*B is that it is lower
% bounded : T * B >= 1. This constraint, known as the HEISENBERG-GABOR 
% INEQUALITY, illustrates the fact that a signal can not have 
% simultaneously an arbitrarily small support in time and in frequency.
% If we consider a Gaussian signal,

sig=amgauss(256); 
subplot(211); plot(real(sig)); axis([1 256 0 1]); grid;
xlabel('Time'); ylabel('Real part'); title('Signal in time');
dsp=fftshift(abs(fft(sig)).^2);
subplot(212); plot((-128:127)/256,dsp); grid;
xlabel('Normalized frequency'); ylabel('Squared modulus'); 
title('Energy spectrum');
[tm,T]=loctime(sig); 
[fm,B]=locfreq(sig);
[T,B,T*B]

% we can see that it minimizes the time-bandwidth product, and thus is 
% the most concentrated signal in the time-frequency plane.
%
% Press any key to continue...
 
pause; clc;

% Instantaneous frequency and group delay
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% The instantaneous frequency, defined for any analytic signal xa(t) as 
% the derivative of its phase, if(t) = 1/(2pi) d arg{xa(t)} / dt, can
% be a good solution to describe a signal simultaneously in time and in 
% frequency :

sig=fmlin(256); t=(3:256); clf;
ifr=instfreq(sig); plotifl(t,ifr'); grid;
axis([1 256 0 0.5]); xlabel('Time'); ylabel('Normalized frequency'); 
title('Instantaneous frequency estimation');
 
% As we can see from this plot, the instantaneous frequency shows with
% success the local frequency behavior as a function of time. 
%
% Press any key to continue...
 
pause;

% In a dual way, the local time behavior as a function of frequency can 
% be described by the GROUP DELAY : 
%	tx(nu) = -1/(2*pi) * d arg{Xa(nu)}/d nu.
% This quantity measures the average time arrival of the frequency nu. 
% For example, with signal sig of the previous example, we obtain :

fnorm=0:.05:.5; gd=sgrpdlay(sig,fnorm); plot(gd,fnorm); grid;
xlabel('Time'); ylabel('Normalized frequency'); 
title('Group delay estimation'); axis([1 256 0 0.5]);
 
% Press any key to continue...
 
pause; clc;

% Be careful of the fact that in general, instantaneous frequency and 
% group delay define two different curves in the time-frequency plane. 
% They are approximatively identical only when the time-bandwidth product 
% TB is large. To illustrate this point, let us consider a simple example.
% We calculate the instantaneous frequency and group delay of two signals, 
% the first one having a large TB product, and the second one a small TB
% product:

t=2:255; 
sig1=amgauss(256,128,90).*fmlin(256,0,0.5);
[tm,T1]=loctime(sig1); [fm,B1]=locfreq(sig1); T1*B1
ifr1=instfreq(sig1,t); f1=linspace(0,0.5-1/256,256);
gd1=sgrpdlay(sig1,f1); subplot(211); plot(t,ifr1,'*',gd1,f1,'-')
axis([1 256 0 0.5]); grid; xlabel('Time'); 
ylabel('Normalized frequency'); 

sig2=amgauss(256,128,30).*fmlin(256,0.2,0.4);
[tm,T2]=loctime(sig2); [fm,B2]=locfreq(sig2); T2*B2
ifr2=instfreq(sig2,t); f2=linspace(0.2,0.4,256);
gd2=sgrpdlay(sig2,f2); subplot(212); plot(t,ifr2,'*',gd2,f2,'-')
axis([1 256 0.2 0.4]); grid; xlabel('Time'); 
ylabel('Normalized frequency'); 
 
% On the first plot, the two curves are almost superimposed (i.e. the
% instantaneous frequency is the inverse transform of the group delay),
% whereas on the second plot, the two curves are clearly different.
%
% Press any key to continue...
 
pause; clc;

% Synthesis of a mono-component non stationary signal
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% One part of the Time-Frequency Toolbox is dedicated to the generation 
% of non stationary signals. In that part, three groups of M-files are 
% available:
%
%	- The first one allows to synthesize different amplitude
% modulations. These M-files begin with the prefix 'am'. 
%	- The second one proposes different frequency modulations.  These
% M-files begin with 'fm'. 
%	- The third one is a set of pre-defined signals. Some of them begin
% with 'ana' because these signals are analytic, other have special names.
% 
% The first two groups of files can be combined to produce a large class of
% non stationary signals, multiplying an amplitude modulation and a 
% frequency modulation. For example, we can multiply a linear frequency 
% modulation by a gaussian amplitude modulation :

fm1=fmlin(256,0,0.5); am1=amgauss(256);
sig1=am1.*fm1; clf; plot(real(sig1)); axis([1 256 -1 1]); 
xlabel('Time'); ylabel('Real part');
 
% By default, the signal is centered on the middle (256/2=128), and its
% spread is T=32. If you want to center it at an other position t0, just
% replace am1 by amgauss(256,t0). 
%
% Press any key to continue...
 
pause; clc; 

% A second example can be to multiply a pure frequency (constant frequency 
% modulation) by a one-sided exponential window starting at t=100 :

fm2=fmconst(256,0.2); am2=amexpo1s(256,100);
sig2=am2.*fm2; plot(real(sig2)); axis([1 256 -1 1]); 
xlabel('Time'); ylabel('Real part');
 
% Press any key to continue...
 
pause; 

% As a third example of mono-component non-stationary signal, we can 
% consider the M-file doppler.m : this function generates a modelization 
% of the signal received by a fixed observer from a moving target emitting 
% a pure frequency.

[fm3,am3]=doppler(256,200,4000/60,10,50);
sig3=am3.*fm3; plot(real(sig3)); axis([1 256 -0.4 0.4]); 
xlabel('Time'); ylabel('Real part');

% This example corresponds to a target (a car for instance) moving 
% straightly at the speed of 50 m/s, and passing at 10 m from the observer
% (the radar!). The rotating frequency of the engine is 4000 revolutions 
% per minute, and the sampling frequency of the radar is 200 Hz.
%
% Press any key to continue...
 
pause; clc; 

%   In order to have a more realistic modelization of physical signals, we
% may need to add some complex noise on these signals. To do so, two M-files
% of the Time-Frequency Toolbox are proposed : noisecg.m generates a complex
% white or colored Gaussian noise, and noisecu.m, a complex white uniform 
% noise. For example, if we add complex colored Gaussian noise on the signal
% sig1 with a signal to noise ratio of -10 dB,

noise=noisecg(256,.8);
sign=sigmerge(sig1,noise,-10); plot(real(sign)); 
Min=min(real(sign)); Max=max(real(sign));
xlabel('Time'); ylabel('Real part'); axis([1 256 Min Max]); 

% the deterministic signal sig1 is now almost imperceptible from the noise.
%
% Press any key to continue...
 
pause; clc; 


% Multi-component non stationary signals 
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
% The notion of instantaneous frequency implicitly assumes that, at each
% time instant, there exists only a single frequency component. A dual
% restriction applies to the group delay : the implicit assumption is that
% a given frequency is concentrated around a single time instant. Thus, if
% these assumptions are no longer valid, which is the case for most of the
% multi-component signals, the result obtained using the instantaneous
% frequency or the group delay is meaningless.
%
% For example, let's consider the superposition of two linear frequency 
% modulations :

N=128; x1=fmlin(N,0,0.2); x2=fmlin(N,0.3,0.5);
x=x1+x2;

% At each time instant t, an ideal time-frequency representation should
% represent two different frequencies with the same amplitude. The results
% obtained using the instantaneous frequency and the group delay are of
% course completely different, and therefore irrelevant :

ifr=instfreq(x); subplot(211); plot(ifr);
xlabel('Time'); ylabel('Normalized frequency'); axis([1 N  0 0.5]);
fnorm=0:0.01:0.5; gd=sgrpdlay(x,fnorm); subplot(212); plot(gd,fnorm);
xlabel('Time'); ylabel('Normalized frequency'); axis([1 N  0 0.5]);
 
% So these one-dimensional representations, instantaneous frequency and 
% group delay, are not sufficient to represent all the non stationary 
% signals. A further step has to be made towards two-dimensional mixed 
% representations, jointly in time and in frequency. 
%
% Press any key to continue...
 
pause; clc; 

% To have an idea of what can be made with an time-frequency decomposition,
% let's anticipate the following and have a look at the result obtained 
% with the Short Time Fourier Transform :

tfrstft(x); 

% Here two 'time-frequency components' can be clearly seen, located around
% the locus of the two frequency modulations.
%
% Press any key to end this demonstration.

pause;
echo off

