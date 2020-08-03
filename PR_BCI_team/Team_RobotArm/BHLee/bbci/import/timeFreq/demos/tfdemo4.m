%TFDEMO4 Cohen's class time-frequency distributions.
%	Time-Frequency Toolbox demonstration.
%
%	See also TFDEMO.

%	O. Lemoine - May 1996. 
%	Copyright (c) CNRS.

clc; zoom on; clf; 
echo on;

% The Wigner-Ville distribution
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% A time-frequency energy distribution which is particularly interesting 
% is the Wigner-Ville distribution (WVD),which satisfies a large number 
% of desirable mathematical properties. Let us see what we obtain on two 
% particular synthetic signals :
% - the first signal is the academic linear chirp signal :

sig=fmlin(128);

% If we choose a 3-dimension plot to represent it, we can see that the WVD
% can take negative values, and that the localization obtained in the
% time-frequency plane for this signal is almost perfect.

tfrwv(sig);
% Press any key to continue...
 
pause; clc;

% - the second one illustrates the Doppler effect, which expresses the 
% dependence of the frequency received by an observer from a transmitter
% on the relative speed between the observer and the transmitter : 

[fm,am,iflaw]=doppler(128,50,13,10,200);
sig=am.*fm;
tfrwv(sig);
% Looking at this time-frequency distribution, we notice that the energy is
% not distributed as we could expect for this signal. Although the signal
% term is well localized in the time-frequency plane, numerous other terms
% (the interference terms, due to the bilinearity of the WVD) are present at
% positions in time and frequency where the energy should be null. 
%
% Press any key to continue...
 
pause; clc; close

% Interference geometry of the WVD
%""""""""""""""""""""""""""""""""""
% The rule of interference construction of the WVD can be summarized as
% follows : two points of the time-frequency plane interfere to create a
% contribution on a third point which is located at their geometrical
% midpoint. Besides, these interference terms oscillate perpendicularly to
% the line joining the two points interfering, with a frequency
% proportional to the distance between these two points.
%  This can be seen on the following example : we consider two atoms in
% the time-frequency plane, analyzed by the WVD, whose relative distance 
% is increasing from one realization to the other, and then decreasing. 
% The WVDs were calculated and saved on the file movieat.mat. We load them 
% and run the sequence using the function movie :

load movwv2at
pause;
movie(M,5);

% We can notice, from this movie, the evolution of the interferences
% when the distance between the two interfering terms changes, and in
% particular the change in the period and the direction of the oscillations.
%
% Press any key to continue...
 
pause; clc;
 
% The pseudo-WVD
%""""""""""""""""
%   As the analyzed signal is not known from -infinity to +infinity in
% pratice, one often consider a windowed version of the WVD, called the
% pseudo-WVD. The time-windowing operated has the effect of smoothing the
% WVD in frequency. Thus, because of their oscillating nature, the 
% interferences will be attenuated in the pseudo-WVD compared to the WVD.
% However, the consequence of this improved readability is that many 
% properties of the WVD are lost.
% If we consider a signal composed of four gaussian atoms, each localized
% at a corner of a rectangle,

sig=atoms(128,[32,.15,20,1;96,.15,20,1;32,.35,20,1;96,.35,20,1]);

% and compute its WVD

tfrwv(sig);
% we can see the four signal terms, along with six interference terms (two of
% them are superimposed). If we now compute the pseudo-WVD,

figure
tfrpwv(sig);
% we can note the important attenuation of the interferences oscillating
% perpendicularly to the frequency axis, and in return the spreading in
% frequency of the signal terms.
%
% Press any key to continue...
 
pause; clc; close;
 
% Importance of the analytic signal
%"""""""""""""""""""""""""""""""""""
%   Due to the quadratic nature of the WVD, its discrete version may be 
% affected by a spectral aliasing, in particular if the signal x is 
% real-valued and sampled at the Nyquist rate. A solution to this problem 
% consists in using the analytic signal. Indeed, as its bandwidth is half the
% one of the real signal, the aliasing will not take place in the useful
% spectral domain [0,1/2] of this signal. This solution presents a second
% advantage : since the spectral domain is divided by two, the number of
% components in the time-frequency plane is also divided by two. 
% Consequently, the number of interference terms decreases significantly. 
% Here is an illustration : we first consider the WVD of the real part of 
% a signal composed of two atoms :

sig=atoms(128,[32,0.15,20,1;96,0.32,20,1]);
tfrwv(real(sig));
% We can see that four signal terms are present instead of two, due to the
% spectral aliasing. Besides, because of the components located at negative
% frequencies (between -1/2 and 0), additional interference terms are
% present. 
%
% Press any key to continue...

pause; 

% If we now consider the WVD of the same signal, but in its complex
% analytic form,

tfrwv(sig);
% the aliasing effect has disappeared, as well as the terms corresponding to
% interferences between negative- and positive- frequency components.
%
% Press any key to continue...

pause; clc; 

% The Cohen's class
%~~~~~~~~~~~~~~~~~~~
%  The Cohen's class gather all the time-frequency energy distributions which
% are covariant by translations in time and in frequency. It can be expressed
% as a 2-D correlation between a function PI(t,nu) and the WVD of the 
% analyzed signal. The WVD is the element of the Cohen's class for which PI 
% is a double Dirac, and the spectrogram is th element for which PI is the 
% WVD of the short time window h. We consider in the following other elements
% of this class
% 
% The smoothed pseudo-WVD
%"""""""""""""""""""""""""
% If we consider a separable smoothing function PI(t,nu)=g(t)H(-nu) (where 
% H(nu) is the Fourier transform of a smoothing window h(t)), we allow a 
% progressive and independent control, in both time and frequency, of the 
% smoothing applied to the WVD. The obtained distribution is known as the 
% smoothed-pseudo WVD. 
% For example,let's consider a signal composed of two components : the first 
% one is a complex sinusoid (normalized frequency 0.15) and the second one 
% is a Gaussian signal shifted in time and frequency :  

sig=sigmerge(fmconst(128,.15),amgauss(128).*fmconst(128,0.4),5);
 
% If we display the WVD, the pseudo-WV and the smoothed-pseudo-WVD of it,

tfrwv(sig);  
figure; tfrpwv(sig); 
figure; tfrspwv(sig);
% we can make the following remarks : from the WVD, we can see the two
% signal terms located at the right positions in the time-frequency plane,
% as well as the interference terms between them. As these interference 
% terms oscillate globally perpendicularly to the time-axis, the frequency
% smoothing done by the pseudo-WVD degrades the frequency resolution without
% really attenuating the interferences. On the other hand, the time-smoothing
% carried out by the smoothed-pseudo-WVD considerably reduces these
% interferences.
%
% Press any key to continue...

pause; clc; close; close; close

%   An interresting property of the smoothed-pseudo WVD is that it allows a
% continuous passage from the spectrogram to the WVD, under the condition
% that the smoothing functions g and h are gaussian. This is clearly 
% illustrated by the function movsp2wv.m, which considers different 
% transitions, on a signal composed of four atoms :
	
load movsp2wv
pause
movie(M,5);

% This movie shows the effect of a (time/frequency) smoothing on the
% interferences and on the resolutions : the WVD gives the best resolutions
% (in time and in frequency), but presents the most important interferences,
% whereas the spectrogram gives the worst resolutions, but with nearly no
% interferences ; and the smoothed-pseudo WVD allows to choose the best
% compromise between these two extremes.
%
% Press any key to continue...

pause; clc; 

% The narrow-band ambiguity function
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%  The narrow-band ambiguity function, often used in radar signal processing,
% is the two-dimensional Fourier transform of the WVD. This property can be 
% used to attenuate some of the interference terms. Indeed, in the case of 
% a multi-component signal, the elements of the AF corresponding to the 
% signal components (denoted as the AF-signal terms) are mainly located 
% around the origin, whereas the elements corresponding to interferences 
% between the signal components (AF-interference terms) appear at a distance
% from the origin which is proportional to the time-frequency distance
% between the involved components. This can be noticed on a simple example :
% We apply consider a signal composed of two linear FM signals with 
% gaussian amplitudes :

N=64; sig1=fmlin(N,0.2,0.5).*amgauss(N);
sig2=fmlin(N,0.3,0).*amgauss(N);
sig=[sig1;sig2]; 

% Let us first have a look at the WVD :

tfrwv(sig);
% We have two distinct signal terms, and some interferences oscillating in
% the middle. 
%
% Press any key to continue...

pause;

% If we look at the ambiguity function of this signal,

clf; ambifunb(sig);

% we have around the origin (in the middle of the image) the AF-signal 
% terms, whereas the AF-interference terms are located away from the origin.
%
% Press any key to continue...

pause; clc;

% Other important energy distributions
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%
% The Rihaczek and Margenau-Hill distributions
% """""""""""""""""""""""""""""""""""""""""""""
%  The Rihaczek distribution, defined as
%	Rx(t,nu)=x(t)X*(nu)e^{-j2pi nu t},
% is a complex energy density at point (t,nu). This distribution, which 
% corresponds to the element of the Cohen's class for which 
% f(xi,tau)=e^{jpi xi tau}, verifies many good properties. However, it is 
% complex valued, which can be awkward in practice. The real part of the 
% Rihaczek distribution is also a time-frequency distribution of the 
% Cohen's class (f(xi,tau)=cos(pi xi tau)), known as the Margenau-Hill
% distribution
%   The interference structure of the Rihaczek and Margenau-Hill
% distributions is different from the Wigner-Ville one : the interference
% terms corresponding to two points located on (t1,nu1) and (t2,nu2) are
% positioned at the coordinates (t1,nu2) and (t2,nu1). This can be seen on
% the following example :

sig=atoms(128,[32,0.15,20,1;96,0.32,20,1]);
tfrmh(sig);
% Thus, the use of the Rihaczek (or Margenau-Hill) distribution for signals
% composed of multi-components located at the same position in time or in
% frequency is no advised, since the interference terms will then be
% superposed to the signal terms.
%
% Press any key to continue...

pause; clc; close

% The Choi-Williams distribution
% """""""""""""""""""""""""""""""
%  An example of reduced interference distribution is given by the
% Choi-Williams distribution, defined as
% CWx(t,nu)=sqrt(2/pi)\int\int sigma/|tau| e^{-2sigma^2(s-t)^2/tau^2} 
%	    x(s+tau/2)x*(s-tau/2) e^{-j2pi nu tau} ds dtau 
% Note that when sigma->+infty, we obtain the WVD. Inversely, the smaller
% sigma, the better the reduction of the interferences. 
%   The "cross"-shape of the parametrization function of the Choi-Williams
% distribution implies that the efficiency of this distribution strongly
% depends on the nature of the analyzed signal. For instance, if the signal
% is composed of synchronized components in time or in frequency, the
% Choi-Williams distribution will present strong interferences. This can be
% observed on the following example : we analyze four gaussian atoms
% positionned at the corners of a rectangle rotating around the center of 
% the time-frequency plane : 

load movcw4at
pause
movie(M,5);

% When the time/frequency supports of the atoms overlap, some 
% AF-interference terms are not be completly attenuated (those present 
% around the axes of the ambiguity plane), and the efficiency of the 
% distribution is quite poor. 
%
% Press any key to continue...

pause; clc;

% Comparison of the parametrization functions
% """"""""""""""""""""""""""""""""""""""""""""
%  To illustrate the differences between some of the presented
% distributions, we represent their weighting (parametrization) function in
% the ambiguity plane, along with the result obtained by applying them on a
% two-component signal embedded in white gaussian noise : the signal is the
% sum of two linear FM signals, the first one with a frequency going from
% 0.05 to 0.15, and the second one from 0.2 to 0.5. The signal to noise 
% ratio is 10 dB.

% On the left-hand side of the figures, the parametrization functions are
% represented in a schematic way by the bold contour lines (the weighting
% functions are mainly non-zeros inside these lines), superimposed to the
% ambiguity function of the signal. The AF-signal terms are in the middle of
% the ambiguity plane, whereas the AF-interference terms are distant from the
% center. On the right-hand side, the corresponding time-frequency
% distributions are represented.

paramfun

% From these plots, we can conclude that the ambiguity plane is very
% enlightening with regard to interference reduction in the case of
% multicomponent signals. On this example, we notice that the
% smoothed-pseudo-WVD is a particularly convenient and versatile
% candidate. This is due to the fact that we can adapt independently the
% time-width and frequency-width of its weighting function. But in the
% general case, it is interesting to have several distributions at our
% disposal since each one is well adapted to a certain type of
% signal. Besides, for a given signal, as a result of the different
% interference geometries, these distributions offer complementary
% descriptions of this signal.
%
% Press any key to end this demonstration

pause; 
echo off

close; clc; clf