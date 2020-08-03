%TFDEMO6 Reassigned time-frequency distributions.
%	Time-Frequency Toolbox demonstration.
%
%	See also TFDEMO.

%	O. Lemoine - August 1996. 
%	Copyright (c) CNRS.

clc; zoom on; clf; 
echo on;

 
% The reassignment of the spectrogram
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   The reassignment method, originally introduced in an attempt to
% improve the spectrogram, moves each value of the spectrogram computed 
% at any point (t,nu) to another point which is the center of gravity 
% of the signal energy distribution around (t,nu). 
%   Let us have a look at the readability improvement obtained by the 
% reassigned spectrogram on an example of multicomponent signal ; the 
% result is compared to the "ideal" representation based on the knowledge 
% of the instantaneous frequency law of each component :

N=128; [sig1 ifl1]=fmsin(N,0.15,0.45,100,1,0.4,-1);
[sig2 ifl2]=fmhyp(N,[1 .5],[32 0.05]);
sig=sig1+sig2;
tfrideal([ifl1 ifl2]);
figure; tfrrsp(sig);
% The improvement given by the reassignment method is obvious : compared to
% the spectrogram, the two components obtained with the reassigned spectrogram 
% are much better localized and almost perfectly concentrated, and there are 
% very few cross-terms.
%
% Press any key to continue...
 
pause; clc; close;

% Reassignment of other quadratic time-frequency distributions
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
%   The reassignment method has been extended to many other time-frequency
% distributions, and in particular to the distributions of the Cohen's class
% and of the affine class. We present here some examples, on a signal composed
% of 3 components : a sinusoidal frequency modulation followed by a pure tone
% together with a linear chirp : 

[sig1 ifl1]=fmsin(60,0.15,0.35,50,1,0.35,1);
[sig2 ifl2]=fmlin(60,0.3,0.1);
[sig3 ifl3]=fmconst(60,0.4);
sig=[sig1;zeros(8,1);sig2+sig3];
iflaw=zeros(128,2);
iflaw(:,1)=[ifl1;NaN*ones(8,1);ifl2];
iflaw(:,2)=[NaN*ones(68,1);ifl3];
% We first plot the instantaneous frequency laws (obtained by tfrideal) 
% and the WVD of this signal :

tfrideal(iflaw);
figure; tfrwv(sig);

% With the WVD, the signal components are well localized, but the numerous
% cross-terms make the figure hardly readable. 
%
% Press any key to continue...
 
pause; clc; 

% If we now consider the smoothed pseudo-WVD and its reassigned version 

tfrrspwv(sig);
% we can see that the smoothing done by the SPWVD almost completly suppress
% the cross terms, but the signal components localization becomes
% coarser. The improvement given by the reassignment method is obvious : all
% components are much better localized, leading to a nearly ideal
% representation.
%
% Press any key to continue...
 
pause; clc; 

% The next distributions we consider are the scalogram and the Morlet 
% scalogram 

figure(1); tfrrsp(sig);
figure(2); tfrrmsc(sig);
% These two distributions present nearly no cross terms, except at
% the bottom of the sinusoid and around time t=64. But the time and
% frequency resolutions are not good, especially at low frequencies in the
% case of the scalogram. The reassignment method improves considerably these
% localizations, and the reassigned spectrogram is even perfectly
% concentrated for the chirp components. The result obtained with the
% modified scalogram is less good, especially at low frequencies where the
% time-resolution is really inadequate. 
%
% Press any key to continue...
 
pause; clc; 

% Finally, we represent the pseudo-Page and the pseudo Margenau-Hill
% distributions with their reassigned version :  

figure(1); tfrrppag(sig);
figure(2); tfrrpmh(sig);
% These representations (before reassignment) are hardly readable since some
% cross-terms are superimposed on the signal components. Their modified
% versions give much better localized signal components, but less
% concentrated than in the case of the spectrogram or the SPWVD.
%
% Press any key to continue...

pause; clc; close

% Connected approaches
%~~~~~~~~~~~~~~~~~~~~~~
%  Connections of the reassignment method has been found with other
% techniques which extract relevant informations from the time-frequency
% plane. 
%
% Friedman's instantaneous frequency density
%""""""""""""""""""""""""""""""""""""""""""""
%  A first example is the instantaneous frequency density : so as to take
% advantage of the phase structure of the short-time Fourier transform
% (STFT), Friedman simply computed at each time t the histogram of the
% frequency displacements of the spectrogram. The resulting time-frequency
% representation is no more an energy distribution, and could be derived 
% as well from any other reassigned distribution.
%  Here is an example of this instantaneous frequency density obtained with
% the pseudo-WVD of the previous signal

t=1:2:127; [tfr,rtfr,hat]=tfrrpwv(sig,t);
friedman(tfr,hat,t,'tfrrpwv',1);
% Although some cross terms are still present, the localization of the
% components is quite good, especially for the chirp components.
%
% Press any key to continue...

pause; clc; 

% Extraction of ridges and skeleton
%"""""""""""""""""""""""""""""""""""
%  This method extracts from any reassigned time-frequency distribution
% (given its displacement operators) some particular sets of curves 
% deduced from the stationary points of their phase.
%  For example, let us extract the ridges from the spectrogram of the previous
% signal :

[tfr,rtfr,hat]=tfrrsp(sig); 
ridges(tfr,hat);
% The result is interesting : apart from some ``gaps'' present in particular
% on the sinusoidal frequency modulation, this method concentrates and
% localizes nearly ideally the signal in the time-frequency plane, even when
% there are two components present at the same time (or at the same
% frequency).


echo off

