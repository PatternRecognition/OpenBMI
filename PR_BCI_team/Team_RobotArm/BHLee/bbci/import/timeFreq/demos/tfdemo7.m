%TFDEMO7 Extraction of information from the time-frequency plane
%	Time-Frequency Toolbox demonstration.
%
%	See also TFDEMO.

%	O. Lemoine - august 1996. 
%	Copyright (c) CNRS.

clc; clf; zoom on; set(gca,'visible','off');
echo on;

% Information from the interferences
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%  The interference terms present in any quadratic time-frequency
% representation, even if they disturb the readability of the
% representation, contain some information about the analyzed
% signal. The precise knowledge of their structure and construction rule
% is useful to interpret the information that they contain. 
% 
%   For instance, the interference terms contain some information about the
% phase of a signal. Let us consider the pseudo WVD of the superposition of
% two constant frequency modulations, with a phase shift between the two
% sinusoids. If we compare the pseudo WVD for different phase shifts, we can
% observe a time-sliding of the oscillating interferences :

pause;
load movpwdph
movie(M,10);

% Each snapshot corresponds to the pseudo WVD with a different phase
% shift between the two components. 
% 
% Press any key to continue...
 
pause; 

%   A second example of phase's signature is given by the influence of a
% jump of phase in a signal analyzed by the (pseudo) Wigner-Ville
% distribution : for instance, if we consider a constant frequency
% modulation presenting a jump of phase in its middle : 

pause; 
load movpwjph
movie(M,10);

% the pseudo WVD presents a pattern around the jump position which is
% all the more important since this jump of phase is close to pi. This
% characteristic can be exploited to detect a jump of phase in a signal.
%
% Press any key to continue...
 
close; pause; clc; 
 

% Renyi information
%~~~~~~~~~~~~~~~~~~~
%  Another interresting information that one may need to know about an
% observed non stationary signal is the number of elementary signals
% composing this observation. Third order Renyi information is a possible
% solution for measuring this information. 
%  This can be observed by considering the WVD of one, two and then
% four elementary atoms, and then by applying the Renyi information on
% them :

sig=atoms(128,[64,0.25,20,1]); 
[TFR,T,F]=tfrwv(sig);
R1=renyi(TFR,T,F)
pause;

sig=atoms(128,[32,0.25,20,1;96,0.25,20,1]); 
[TFR,T,F]=tfrwv(sig);
R2=renyi(TFR,T,F)
pause;

sig=atoms(128,[32,.15,20,1;96,.15,20,1;32,.35,20,1;96,.35,20,1]);  
[TFR,T,F]=tfrwv(sig);
R3=renyi(TFR,T,F)

% We can see that if R is set to 0 for one elementary atom by
% subtracting R1, we obtain a result close to 1 for 2 atoms
% (R2-R1=0.99) and close to 2 for 4 atoms ( R3-R1=2.01). If the components 
% are less separated in the time-frequency plane, the information measure
% will be affected by the overlapping of the components or by the 
% interference terms between them. In particular, it is possible to show that
% the Renyi information measure provides a good indication of the time
% separation at which the atoms are essentially resolved, more precise than
% does the time-bandwidth product.
%
% Press any key to continue...
 
pause; clc; clf; 
 
% Time-frequency analysis : help to decision
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%  The Wigner-Hough transform
% """"""""""""""""""""""""""""
%  Applying the Hough transform (detection of lines in images) to the WVD
% of a linear chirp signal leads to a new tranform, the Wigner-Hough transform,
% which ideally concentrates this kind of signal in a parameter space. Then,
% by comparing this 2-dimensional statistic to a threshold, we obtain the 
% asymptotically optimal detector (in the case of white gaussian noise). 
% Moreover, the coordinates of the peak detected give estimators of the chirp
% parameters which asymptotically reach the Cramer-Rao lower bounds.   
% 
%   Here is an illustration of this decision test : first, we consider
% a linear chirp signal embedded in a white gaussian noise, with a 1 dB 
% signal-to-noise ratio :

N=64; sig=sigmerge(fmlin(N,0,0.3),noisecg(N),1);

% Now, if we analyze it with the WVD followed by the Hough transform,

tfr=tfrwv(sig); contour(tfr,5); grid; 
htl(tfr,N,N,1);

% we obtain, in the parameters' space (rho,theta), a peak representing
% the chirp signal, significantly more energetic than the other peaks
% corresponding to the noise. The decision test is then very simple : it
% consists in applying a threshold on this representation, positioned
% according to a detection criterion ; if the peak is higher than the
% threshold, then the chirp is said to be present, and the coordinates of
% that peak (hat{rho},hat{theta}) provide estimates of the chirp
% parameters (the change from (hat{rho},hat{theta}) to
% (hat{nu0},hat{beta}) corresponds to the change from polar to
% Cartesian coordinates).
%
% Press any key to continue...
 
pause; clc

%   In the case of a multi-component signal, the problem of interference
% terms appear. However, due to the oscillating structure of these
% terms, the integration operated by the Hough transform on the WVD will 
% attenuate them. This can be observed on the following example :
% we superpose two chirp signals with different initial frequencies and
% sweep rates :

sig=sigmerge(fmlin(N,0,0.4),fmlin(N,0.3,0.5),1);
tfr=tfrwv(sig); contour(tfr,5); grid
htl(tfr,N,N,1);

% We can see that the components are well separated in the parameter
% space, in spite of the use of a nonlinearity in the WHT. Again, the
% coordinates of the two peaks provide estimates of the different
% parameters. 
%
% Press any key to continue...
 
pause; clc

% Analysis of local singularities
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%  Since they are time-dependent in nature, the wavelet-based
% techniques also allow an estimation of the local regularity of a
% signal. 
% 
%   For instance, we consider a 64-points Lipschitz singularity of 
% strength H=0, centered at t0=32,
 
sig=anasing(64);

% and analyze it with the scalogram (Morlet wavelet with half-length=4),

[tfr,t,f]=tfrscalo(sig,1:64,4,0.01,0.5,256,1);

% The time-localization of the singularity can be clearly estimated from
% the scalogram distribution at small scales :

H=holder(tfr,f,1,256,32,1)
 
% This value is a good estimation of H (=0 here)
%
% Press any key to continue...
 
pause;

% If we now consider a singularity of strength H=-0.5,

sig=anasing(64,32,-0.5);
[tfr,t,f]=tfrscalo(sig,1:64,4,0.01,0.5,256,1);

% we notice the different behavior of the scalogram along scales, whose
% decrease is characteristic of the strength H. The estimation of the
% Holder exponent at t=32 gives :

H=holder(tfr,f,1,256,32,1)

% which is close to 0.5.

pause; clf; clc; 

%-----------------------------------------------------------------------
% Thank you for your attention. We hope that you enjoyed this 
% demonstration, and that your understanding in time-frequency analysis 
% has made some progress. Now have a nice time with the Time-Frequency
% Toolbox.  
%-----------------------------------------------------------------------

echo off

