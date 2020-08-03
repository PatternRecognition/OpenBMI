%TFDEMO5 Affine class time-frequency distributions.
%	Time-Frequency Toolbox demonstration.
%
%	See also TFDEMO.

%	O. Lemoine - July 1996. 
%	Copyright (c) CNRS.

clc; zoom on; clf; 
echo on;

% The Affine class : presentation
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% This class gathers all the quadratic time-frequency representations 
% which are covariant by translation in time and dilation. The WVD is
% an element of the affine class, provided that we introduce an 
% arbitrary non-zero frequency nu0, and identify the scale with the 
% inverse of the frequency : a=nu0/nu.
% The choice of an element in the affine class can be reduced to the 
% choice of an affine correlation kernel PI(t,nu). When PI is a 
% two-dimensional low-pass function, it plays the role of an affine
% smoothing function which tries to reduce the interferences generated 
% by the WVD.
%
% The scalogram 
%"""""""""""""""
%  A first example of affine distribution is given by the scalogram,
% which is the squared modulus of the wavelet transform. It is the affine
% counterpart of the spectrogram. As illustrated in the following example,
% the tradeoff between time and frequency resolutions encountered with the
% spectrogram is also present with the scalogram.
%  We analyze a signal composed of two gaussian atoms, one with a low 
% central frequency, and the other with a high one, with the scalogram 
% (Morlet wavelet) :

sig=atoms(128,[38,0.1,32,1;96,0.35,32,1]);
clf; tfrscalo(sig);
% The result obtained brings to the fore dependency, with regard to the 
% frequency, of the smoothing applied to the WVD, and consequently of the
% resolutions in time and frequency.
%
% Press any key to continue...
 
pause; clc; clf; set(gca,'visible','off');
 
% The affine smoothed pseudo Wigner distribution (ASPWVD)
%"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
%  One way to overcome the tradeoff between time and frequency resolutions
% of the scalogram is, as for the smoothed-pseudo-WVD, to use a smoothing
% function which is separable in time and frequency. The resulting
% distribution is called the affine smoothed pseudo WVD. It allows a 
% flexible choice of time and scale resolutions in an independent manner 
% through the choice of two windows g and h. 
%
%  As for the SPWVD, the ASPWVD allows a continuous passage from the 
% scalogram to the WVD, under the condition that the smoothing functions 
% g and h are gaussian. The time-bandwidth product then goes from 1 
% (scalogram) to 0 (WVD), with an independent control of the time and 
% frequency resolutions. This is illustrated in the following example :
	
load movsc2wv
pause
movie(M,5);

% Here again, the WVD gives the best resolutions (in time and in frequency),
% but presents the most important interferences, whereas the scalogram gives
% the worst resolutions, but with nearly no interferences ; and the affine
% smoothed-pseudo WVD allows to choose the best compromise between these two
% extremes.
%
% Press any key to continue...
 
pause; clc; close

% The localized bi-frequency kernel (or affine Wigner) distributions
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%  A useful subclass of the affine class consists in characterization
% functions which are perfectly localized on power laws or logarithmic laws
% in their bi-frequency representation. The corresponding time-scale 
% distributions are known as the localized bi-frequency kernel distributions.
% 
% The Bertrand distribution
%"""""""""""""""""""""""""""
%  If we further impose to these distributions the a priori requirements of
% time localization and unitarity, we obtain the Bertrand distribution. This
% distribution satisfies many properties, and is the only localized
% bi-frequency kernel distribution which localizes perfectly the hyperbolic
% group delay signals. To illustrate this property, consider the signal 
% obtained using the file gdpower.m (taken for k=0), and analyze it with 
% the file tfrbert.m :

sig=gdpower(128);
tfrbert(sig,1:128,0.01,0.22,128,1);
% Note that the distribution obtained is well localized on the hyperbolic
% group delay, but not perfectly : this comes from the fact that the file
% tfrbert.m works only on a subpart of the spectrum, between two bounds fmin
% and fmax.
%
% Press any key to continue...
 
pause; clc;

% The D-Flandrin distribution 
%"""""""""""""""""""""""""""""
%  If we now look for a localized bi-frequency kernel distribution which is
% real, localized in time and which validates the time-marginal property, 
% we obtain the D-Flandrin distribution. It is the only localized 
% bi-frequency kernel distribution which localizes perfectly signals having 
% a group delay in 1/sqrt(nu). This can be illustrated as following :

sig=gdpower(128,1/2);
tfrdfla(sig,1:128,0.01,0.22,128,1);
% Here again, the distribution is almost perfectly localized.
%
% Press any key to continue...
 
pause; clc;

% The active Unterberger distribution
%"""""""""""""""""""""""""""""""""""""
%  Finally, the only localized bi-frequency kernel distribution which
% localizes perfectly signals having a group delay in 1/nu^2 is the active
% Unterberger distribution :

sig=gdpower(128,-1);
tfrunter(sig,1:128,'A',0.01,0.22,172,1);
% Press any key to continue...
 
pause; clc;

% Relation with the ambiguity domain
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%  When the signal under analysis can not be considered as narrow-band
% (i.e. when its bandwidth B is not negligible compared to its central
% frequency nu0), the narrow-band ambiguity function is no longer appropriate
% since the Doppler effect can not be approximated as a frequency-shift. We
% then consider a wide-band ambiguity function (WAF). It corresponds to 
% the wavelet transform of the signal x, whose mother wavelet is the signal
% x itself. It is then an affine correlation function, which measure the 
% similarity between the signal and its translated (in time) and dilated 
% versions. To see how it behaves on a practical example, let us consider an
% Altes signal :
	
sig=altes(128,0.1,0.45);
clf; ambifuwb(sig);

% The WAF is maximum at the origin of the ambiguity plane.  
%
% Press any key to continue...
 
pause; clc
  
% Interference structure
%~~~~~~~~~~~~~~~~~~~~~~~~
%  The interference structure of the localized bi-frequency kernel 
% distributions can be determined thanks to the following geometric 
% argument : two points (t1,nu1) and (t2,nu2) belonging to the trajectory 
% on which a distribution is localized interfere on a third point 
% (ti,nui) which is necessarily located on the same trajectory.
%  To illustrate this interference geometry, let us consider the case of a
% signal with a sinusoidal frequency modulation :

[sig,ifl]=fmsin(128);

% The file plotsid.m allows one to construct the interferences of an affine
% Wigner distribution perfectly localized on a power-law group-delay
% (specifying k), for a given instantaneous frequency law (or the
% superposition of different instantaneous frequency laws). For example, if
% we consider the case of the Bertrand distribution (k=0),

plotsid(1:128,ifl,0);

% we obtain an interference structure completely different from the one
% obtained for the Wigner-Ville distribution (k=2) :
%
% press any key to continue...
 
pause;

plotsid(1:128,ifl,2);

% For the active Unterberger distribution (k=-1), the result is the
% following : 
%
% press any key to continue...
 
pause;

plotsid(1:128,ifl,-1);
 
% Press any key to continue...
 
pause; clc

% The pseudo affine Wigner distributions
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   The affine Wigner distributions show great potential as flexible
% tools for time-varying spectral analysis. However, as some distributions of
% the Cohen's class, they present two major practical limitations : first the
% entire signal enters into the calculation of these distributions at every
% point (t,nu), and second, due to their nonlinearity, interference
% components arise between each pair of signal components. To overcome these
% limitations, a set of (smoothed) pseudo affine Wigner distributions has
% been introduced.
%  Here are two examples of such distributions, analyzed on a real 
% echolocation signal from a bat :

load bat; N=128;
sig=hilbert(bat(801:7:800+N*7)');

% The affine smoothed pseudo Wigner distribution 
%------------------------------------------------

figure(1); tfrwv(sig); 
figure(2); tfrspaw(sig,1:N,2,24,0,0.1,0.4,N,1); 

% On the left, the WVD presents interference terms because of the
% non-linearity of the frequency modulation. On the right, the affine
% frequency smoothing operated by the affine smoothed pseudo Wigner
% distribution almost perfectly suppressed the interference terms.
%
% Press any key to continue...
 
pause; clc

% The pseudo Bertrand distribution
%----------------------------------

figure(1); tfrbert(sig,1:N,0.1,0.4,N,1);
figure(2); tfrspaw(sig,1:N,0,32,0,0.1,0.4,N,1); 

% The first plot represents the Bertrand distribution. The approximate
% hyperbolic group delay law of the bat signal explains the good result
% obtained with this distribution (compared to the WVD). However, it
% remains some interference terms, which are almost perfectly canceled
% on the second plot (pseudo Bertrand distribution).
%
% Press any key to end this demonstration

pause; close;
echo off

