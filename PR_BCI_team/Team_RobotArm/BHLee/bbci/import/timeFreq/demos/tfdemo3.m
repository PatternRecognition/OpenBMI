%TFDEMO3 Demonstration on linear time-frequency representations.  	 
%	Time-Frequency Toolbox demonstration.
%
%	See also TFDEMO.

%	O. Lemoine - May 1996. 
%	Copyright (c) CNRS.

clc; zoom on; clf; 
echo on;

% The Short-Time Fourier Transform
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% In order to introduce time-dependency in the Fourier transform, a simple
% and intuitive solution consists in pre-windowing the signal x(u) around a
% particular time t, calculating its Fourier transform, and doing that for
% each time instant t. The resulting transform is called the Short-Time 
% Fourier Transform (STFT).
%
% Let us have a look at the result obtained by applying the STFT on a
% speech signal. The signal we consider contains the word 'GABOR' recorded 
% on 338 points with a sampling frequency of 1 kHz (with respect to the 
% Shannon criterion).

load gabor; time=0:337; 
subplot(211); plot(time,gabor); xlabel('Time [ms]'); grid

% Now let us have a look at the Fourier transform of it :

dsp=fftshift(abs(fft(gabor)).^2); subplot(212); 
freq=(-169:168)/338*1000; plot(freq,dsp); xlabel('Frequency [Hz]'); grid

% We can not say from this representation what part of the word is
% responsible for that peak around 140 Hz. 
%
% Press any key to continue...
 
pause; clc;
 
% Now if we look at the squared modulus of the STFT of this signal, 
% using a hamming analysis window of 85 points, we can see some interesting
% features (the time-frequency matrix is loaded from the MAT-file because 
% it takes a long time to be calculated ; we represent only the frequency 
% domain where the signal is present) :
		
clf; contour(time,(0:127)/256*1000,log10(tfr)); grid
xlabel('Time [ms]'); ylabel('Frequency [Hz]'); 
title('Squared modulus of the STFT of the word GABOR');

% The first pattern in the time-frequency plane, located between 30ms and
% 60ms, and centered around 150Hz, corresponds to the first syllable
% 'GA'. The second pattern, located between 150ms and 250ms, corresponds to
% the last syllable 'BOR', and we can see that its mean frequency is
% decreasing from 140Hz to 110Hz with time. Harmonics corresponding to these
% two fondamental signals are also present at higher frequencies, but with a
% lower amplitude.
%
% Press any key to continue...
 
pause; clc;
 
% To illustrate the tradeoff which exists for the STFT between time and 
% frequency resolutions, whatever is the short time analysis window h, we 
% consider two extreme cases : 
% - the first one corresponds to a perfect time resolution : the analysis 
% window h(t) is chosen as a Dirac impulse :

sig=amgauss(128).*fmlin(128); h=1;
tfrstft(sig,1:128,128,h);

% The signal is perfectly localized in time (a section for a given 
% frequency of the squared modulus of the STFT corresponds exactly to the 
% squared modulus of the signal), but the frequency resolution is null.     
%
% Press any key to continue...
 
pause; 

% - the second is that of perfect frequency resolution , obtained with a
% constant window :

h=ones(127,1);
tfrstft(sig,1:128,128,h);

% Here the STFT reduces to the Fourier transform (except on the sides, 
% because of the finite length of h), and does not provides any time 
% resolution.  
%    
% Press any key to continue...
 
pause; clc

% To illustrate the influence of the shape and length of the analysis
% window h, we consider two transient signals having the same gaussian
% amplitude and constant frequency, with different arrival times :

sig=atoms(128,[45,.25,32,1;85,.25,32,1]);

% Here is the result obtained with a Hamming analysis window of 65 
% points :

h=window(65,'hamming');
tfrstft(sig,1:128,128,h);

% The frequency-resolution is very good, but it is almost impossible to
% discriminate the two components in time. 
%    
% Press any key to continue...
 
pause; clc

% If we now consider a short Hamming window of 17 points,

h=window(17,'hamming');
tfrstft(sig,1:128,128,h);

% the frequency resolution is poorer, but the time-resolution is 
% sufficiently good to distinguish the two components. 
%    
% Press any key to continue...
 
pause; clc; clf

% The Gabor Representation 
%~~~~~~~~~~~~~~~~~~~~~~~~~~
% The reconstruction (synthesis) formula of the STFT given in the 
% discrete case defines the Gabor representation. Let us consider the 
% Gabor coefficients of a linear chirp of N1=256 points at the critical 
% sampling case, and for a gaussian window of Ng=33 points :

N1=256; Ng=33; Q=1; % degree of oversampling.
sig=fmlin(N1); g=window(Ng,'gauss'); g=g/norm(g);
[tfr,dgr,h]=tfrgabor(sig,16,Q,g);

% (tfrgabor generates as first output the squared modulus of the Gabor
% representation, as second output the complex Gabor representation, and 
% as third output the biorthonormal window). When we look at the
% biorthonormal window h,

plot(h); axis([1 256 -0.3 0.55]); grid; title('Biorthonormal window'); 

% we can see how "bristling" this function is. 
%    
% Press any key to continue...
 
pause; clc

% The corresponding Gabor decomposition contains all the information about 
% sig, but is not easy to interpret :

t=1:16; f=linspace(0,0.5,8); imagesc(t,f,tfr(1:8,:));  grid
xlabel('Time'); ylabel('Normalized frequency'); axis('xy'); 
title('Squared modulus of the Gabor coefficients');

% Press any key to continue...
 
pause;

% If we now consider a degree of oversampling of Q=4 (there are four times
% more Gabor coefficients than signal samples), the biorthogonal function is
% smoother (the bigger Q, the closer h from g),

Q=4; [tfr,dgr,h]=tfrgabor(sig,32,Q,g);
plot(h); title('Biorthonormal window'); axis([1 256 -0.01 0.09]); grid; 

% press any key to continue...
 
pause; 

% and the Gabor representation is much more readable :

t=1:32; f=linspace(0,0.5,16); imagesc(t,f,tfr(1:16,:)); axis('xy'); 
xlabel('Time'); ylabel('Normalized frequency');  grid
title('Squared modulus of the Gabor coefficients');

% Press any key to continue...
 
pause; clc; 

% From atomic decompositions to energy distributions
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% The spectrogram
%"""""""""""""""""
% If we consider the squared modulus of the STFT, we obtain a spectral
% energy density of the locally windowed signal x(u) h*(u-t), which 
% defines the spectrogram.
% To illustrate the resolution tradeoff of the spectrogram and its
% interference structure, we consider a two-component signal composed of 
% two parallel chirps :

sig=fmlin(128,0,0.4)+fmlin(128,0.1,0.5);
h1=window(23,'gauss'); figure(1); tfrsp(sig,1:128,128,h1);

h2=window(63,'gauss'); figure(2); tfrsp(sig,1:128,128,h2);

print -deps EPS/At4fig2

% In these two cases, the signals sig1 and sig2 are not sufficiently 
% distant to have distinct terms in the time-frequency plane, whatever the 
% window length is. Consequently, interference terms are present, and 
% disturb the readability of the time-frequency representation. 
%
% Press any key to continue...
 
pause; clc; 

% If we consider more distant components,

sig=fmlin(128,0,0.3)+fmlin(128,0.2,0.5);
h1=window(23,'gauss'); figure(1); tfrsp(sig,1:128,128,h1);
h2=window(63,'gauss'); figure(2); tfrsp(sig,1:128,128,h2);

% the two auto-spectrograms do not overlap and no interference term
% appear. We can also see the effect of a short window (h1) and a long
% window (h2) on the time-frequency resolution. In the present case, the 
% long window h2 is preferable since as the frequency progression is not
% very fast, the quasi-stationary assumption will be correct over h2 (so 
% time resolution is not as important as frequency resolution in this case) 
% and the frequency resolution will be quite good ; whereas if the window 
% is short (h1), the time resolution will be good, which is not very useful, 
% and the frequency resolution will be poor.
%
% Press any key to continue...
 
pause; clc; close;

% The scalogram
%"""""""""""""""
% A similar distribution to the spectrogram can be defined in the wavelet
% case. The squared modulus of the continuous wavelet transform also 
% defines an energy distribution which is known as the scalogram.
% As for the wavelet transform, time and frequency resolutions of the
% scalogram are related via the Heisenberg-Gabor principle : time and
% frequency resolutions depend on the considered frequency. To illustrate
% this point, we represent the scalograms of two different signals. The
% M-file tfrscalo.m generates this representation. The chosen wavelet is a
% Morlet wavelet of 12 points. The first signal is a Dirac pulse at time
% t0=64 :

sig1=anapulse(128);
tfrscalo(sig1,1:128,6,0.05,0.45,64);

% This figure shows that the influence of the signal's behavior around 
% t=t0 is limited to a cone in the time-scale plane (which is more visible 
% if you choose the logarithmic scale is the menu) : it is "very" localized 
% around t0 for small scales (large frequencies), and less and less 
% localized as the scale increases (as the frequency decreases).
%
% Press any key to continue...
 
pause; clc; 

% The second signal is the sum of two sinusoids of different frequencies :

sig2=fmconst(128,.15)+fmconst(128,.35);
tfrscalo(sig2,1:128,6,0.05,0.45,128);
 
% Here again, we notice that the frequency resolution is clearly a function
% of the frequency : it increases with nu.
%
% Press any key to end this demonstration

pause;
echo off

