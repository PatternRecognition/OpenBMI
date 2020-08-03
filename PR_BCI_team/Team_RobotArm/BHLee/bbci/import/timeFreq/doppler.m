function [fm,am,iflaw]=doppler(N,Fs,f0,d,v,t0,c);
%DOPPLER Generate complex Doppler signal.
% 	 [FM,AM,IFLAW]=DOPPLER(N,FS,F0,D,V,T0,C) 
%	 Returns the frequency modulation (FM), the amplitude 
%	 modulation (AM) and the instantaneous frequency law (IFLAW) 
%	 of the signal received by a fixed observer from a moving target 
%	 emitting a pure frequency f0.
%
%	 N  : number of points.  
%	 FS : sampling frequency (in Hertz).  
%	 F0 : target   frequency (in Hertz).  
%	 D  : distance from the line to the observer (in meters).  
%	 V  : target velocity    (in m/s)
%	 T0 : time center                  (default : N/2).  
%	 C  : wave velocity      (in m/s)  (default : 340). 
%	 FM : Output frequency modulation.  
%	 AM : Output amplitude modulation.  
%	 IFLAW : Output instantaneous frequency law.
% 
%	Example: 
%	 N=512; [fm,am,iflaw]=doppler(N,200,65,10,50); 
%	 subplot(211); plot(real(am.*fm)); 
%	 subplot(212); plot(iflaw);
%        [ifhat,t]=instfreq(sigmerge(am.*fm,noisecg(N),15),11:502,10);
%        hold on; plot(t,ifhat,'g'); hold off;
%
%	See also DOPNOISE 

%	F. Auger, July 94, August 95 - O. Lemoine, October 95.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin <= 4),
 error ( 'At least 5 parameters are required' ); 
elseif (nargin == 5),
 t0=N/2; c=340.0;
elseif (nargin == 6),
 c=340.0;
end;

if (N <= 0),
 error ('The signal length N must be strictly positive' );
elseif (d <= 0.0),
 error ('The distance D must be positive' );
elseif (Fs < 0.0),
 error ('The sampling frequency FS must be positive' );
elseif (t0<1) | (t0>N),
 error ('T0 must be between 1 and N');
elseif (f0<0)|(f0>Fs/2),
 error ('F0 must be between 0 and FS/2');
elseif (v<0),
 error ('V must be positive');
else
 tmt0=((1:N)'-t0)/Fs;
 dist=sqrt(d^2+(v*tmt0).^2);
 fm = exp(j*2.0*pi*f0*(tmt0-dist/c));
 if (nargout>=2), 
  if abs(f0)<eps,
   am=0;
  else
   am= 1.0 ./ sqrt(dist); 
  end
 end;
 if (nargout==3), iflaw=(1-v^2*tmt0./dist/c)*f0/Fs; end;
end ;
				   
