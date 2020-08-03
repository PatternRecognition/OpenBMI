function tfrparam(method);
%TFRPARAM Return the paramaters needed to display (or save) this TFR.
%	TFRPARAM(METHOD) returns on the screen the meaning of the
%	parameters P1..P5 used in the files TFRQVIEW, TFRVIEW and TFRSAVE,
%	to view or save a time-frequency representation.
%
%	METHOD : chosen representation (name of the corresponding M-file).   
%
%	Example : 
%	 tfrparam('tfrspwv');
%
%	See also TFRQVIEW, TFRVIEW, TFRPARAM.

%	O. Lemoine, June 1996.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org

method=upper(method);
disp(' ');

if strcmp(method,'TFRWV') | strcmp(method,'TFRMH') | ...
   strcmp(method,'TFRPAGE'),
disp('No parameter needed');

elseif strcmp(method,'TFRPWV'  )|strcmp(method,'TFRPMH'  )| ...
       strcmp(method,'TFRSP'   )|strcmp(method,'TFRPPAGE')| ...
       strcmp(method,'TFRRSP'  )|strcmp(method,'TFRRPPAG')| ...
       strcmp(method,'TFRRPWV' )|strcmp(method,'TFRRPMH' )| ...
       strcmp(method,'TFRSTFT'),
disp('P1 : frequency smoothing window (odd length, column vector)');

elseif strcmp(method,'TFRSPWV' ) | strcmp(method,'TFRMHS' )| ...
       strcmp(method,'TFRRSPWV') | strcmp(method,'TFRRMHS')| ...
       strcmp(method,'TFRRIDBN') | strcmp(method,'TFRZAM' )| ...
       strcmp(method,'TFRBJ'   ) | strcmp(method,'TFRRIDB')| ...
       strcmp(method,'TFRRIDH' ) | strcmp(method,'TFRRIDT'),
disp('P1 : time      smoothing window (odd length, column vector)');
disp('P2 : frequency smoothing window (odd length, column vector)');

elseif strcmp(method,'TFRMMCE'),
disp('P1 : matrix of frequency smoothing windows (each column is a ');
disp('     unit energy smoothing window with odd length)'); 

elseif strcmp(method,'TFRCW' ) | strcmp(method,'TFRBUD'),
disp('P1 : time      smoothing window (odd length, column vector)');
disp('P2 : frequency smoothing window (odd length, column vector)');
disp('P3 : kernel width (strictly positive scalar)');

elseif strcmp(method,'TFRGRD')
disp('P1 : time      smoothing window (odd length, column vector)');
disp('P2 : frequency smoothing window (odd length, column vector)');
disp('P3 : kernel width (strictly positive scalar)');
disp('P4 : dissymmetry ratio (strictly positive scalar)');

elseif strcmp(method,'TFRMSC' ) | strcmp(method,'TFRRMSC' )
disp('P1 : time-bandwidth product of the mother wavelet (positive scalar)');

elseif strcmp(method,'TFRRGAB' )
disp('P1 : length of the gaussian window (positive scalar)');

elseif strcmp(method,'TFRDFLA' ) | strcmp(method,'TFRUNTER' )| ...
       strcmp(method,'TFRBERT'  ),
disp('P1 : number of Mellin points (strictly positive scalar)');
disp('P2 : vector of normalized frequencies (geometrically sampled)');

elseif strcmp(method,'TFRSCALO'),
disp('P1 : half length of the Morlet analyzing wavelet at coarsest scale');
disp('     If P1 = 0, the Mexican hat wavelet is used');
disp('P2 : number of Mellin points (strictly positive scalar)');
disp('P3 : vector of normalized frequencies (geometrically sampled)');

elseif strcmp(method,'TFRASPW'),
disp('P1 : half length of the Morlet analyzing wavelet at coarsest scale');
disp('P2 : half length of the time smoothing window (positive scalar)');
disp('P3 : number of Mellin points (strictly positive scalar)');
disp('P4 : vector of normalized frequencies (geometrically sampled)');

elseif strcmp(method,'TFRSPAW'),
disp('P1 : K variable of the SP affine Wigner distribution (scalar)');
disp('P2 : half length of the Morlet analyzing wavelet at coarsest scale.');
disp('P3 : half length of the time smoothing window (positive scalar)');
disp('P4 : number of Mellin points (strictly positive scalar)');
disp('P5 : vector of normalized frequencies (geometrically sampled)');

elseif strcmp(method,'TFRGABOR'),
disp('P1 : number of Gabor coefficients in time (strictly positive integer)');
disp('P2 : degree of oversampling (strictly positive integer)');
disp('P3 : synthesis window (odd length, column vector)');

end;

disp(' ');