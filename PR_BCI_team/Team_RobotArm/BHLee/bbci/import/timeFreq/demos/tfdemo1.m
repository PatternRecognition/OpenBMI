%TFDEMO1 Introduction to the Time-Frequency Toolbox.

%	O. Lemoine - May 1996. 
%	Copyright (c) CNRS.

clc; zoom on; clf; 
echo on;

% Welcome to the Time-Frequency Toolbox for MATLAB. This demonstration 
% follows the plan of the tutorial, and consider exactly the same examples.
% Therefore, for further information about these illustrations, we advise
% you to refer to the corresponding chapter of the tutorial.  	
%
% Press any key to continue...
 
pause; clc;
 
% First let's create an analytic linear frequency modulated signal, whose 
% normalized frequency is changing from 0 to 0.5 :

sig1=fmlin(128,0,0.5);
plot(real(sig1)); axis([1 128 -1 1]);
xlabel('Time'); ylabel('Real part');
title('Linear frequency modulation'); grid

% From this time-domain representation, it is difficult to say what kind 
% of modulation is contained in this signal.
%
% Press any key to continue...

pause;

% Now let's consider its energy spectrum :
 
dsp1=fftshift(abs(fft(sig1)).^2); 
plot((-64:63)/128,dsp1);        
xlabel('Normalized frequency'); ylabel('Squared modulus');
title('Spectrum'); grid

% We still can not say, from this plot, anything about the evolution in
% time of the frequency content.
%
% In order to have a more informative description of such a signal, it 
% would be better to directly represent their frequency content while 
% still keeping the time description parameter : this is precisely the
% aim of time-frequency analysis. To illustrate this, let's try the
% Wigner-Ville distribution on this signal :
%
% press any key to continue...

pause;  

tfrwv(sig1);

% We can see on this representation that the linear progression of the 
% frequency with time, from 0 to 0.5, is clearly shown.
%
% Press any key to continue...

pause; clc; 

% If we now add some complex white Gaussian noise on this signal, with 
% a 0 dB signal to noise ratio,

sig2=sigmerge(sig1,noisecg(128),0);
Min=min(real(sig2)); Max=max(real(sig2)); 
clf; plot(real(sig2)); axis([1 128 Min Max]);
xlabel('Time'); ylabel('Real part');
title('Linear frequency modulation plus noise'); grid

% press any key to continue...

pause;

% and consider the spectrum of it :

dsp2=fftshift(abs(fft(sig2)).^2); 
plot((-64:63)/128,dsp2);        
xlabel('Normalized frequency'); ylabel('Squared modulus');
title('Spectrum'); grid

% it is worse than before to interpret these plots. On the other hand, the
% Wigner-Ville distribution still show quite clearly the linear progression
% of the frequency with time : 
%
% press any key to continue...

pause;

tfrwv(sig2);

% press any key to continue...

pause; clc; 

% The second example we consider is a bat sonar signal, recorded with a 
% sampling frequency of 230.4 kHz and an effective bandwidth equal to
% [8 kHz, 80 kHz].
%  First, load the signal from the MAT-file bat.mat :

load bat
t0=linspace(0,2500/2304,2500);   
clf; plot(t0,bat); xlabel('Time [ms]');
axis([t0(1) t0(2500) -900 800]); grid; 

% From this plot, we can not say precisely what is the frequency content 
% at each time instant t ; similarly, if we look at its spectrum,
%
% press any key to continue...

pause;
dsp=fftshift(abs(fft(bat)).^2);
f0=(-1250:1249)*230.4/2500;
plot(f0,dsp); xlabel('Frequency [kHz]'); ylabel('Squared modulus');
title('Spectrum'); grid

% we can not say at what time the signal is located around 38 kHz, and at
% what time around 40 kHz. Let us now consider a representation called 
% the pseudo Wigner-Ville distribution, applied on the most interesting 
% part of this signal (this distribution was obtained with the M-file 
% tfrpwv.m, stored in the matrix tfr and saved with the signal in the 
% MAT-file bat.m ; the corresponding time- and freqency- samples t and f 
% where also saved on bat.mat) :
%
% press any key to continue...

pause;
contour(t,f,tfr,5); axis('xy'); 
xlabel('Time [ms]'); ylabel('Frequency [kHz]'); 
title('TFRPWV of a bat signal'); grid

% We then have a nice description of its spectral content varying with 
% time : it is narrow-band signal, whose frequency content is decreasing
% from around 55 kHz to 38kHz, with a non-linear frequency modulation
% (approximately of hyperbolic shape).
%
% press any key to continue...

pause; clc;


% The last introductory example presented here is a transient signal
% embedded in a -5 dB white Gaussian noise. This transient signal is a
% constant frequency modulated by a one-sided exponential amplitude :

trans=amexpo1s(64).*fmconst(64);
sig=[zeros(100,1) ; trans ; zeros(92,1)];
sign=sigmerge(sig,noisecg(256),-5);
Min=min(real(sign)); Max=max(real(sign)); 
subplot(211); plot(real(sign)); axis([1 256 Min Max]);
xlabel('Time'); title('Noisy transient signal'); grid
dsp=fftshift(abs(fft(sign)).^2);
subplot(212); plot((-128:127)/256,dsp); grid
xlabel('Normalized frequency'); title('Energy spectrum');

% From these representations, it is difficult to localize precisely the
% signal in the time-domain as well as in the frequency domain.  Now let us
% have a look at the spectrogram of this signal :
%
% press any key to continue...

pause;
tfrsp(sign);

% the transient signal appears distinctly around the normalized frequency
% 0.25, and between time points 125 and 160.
%
% Press any key to return to the main menu.

pause;
echo off

