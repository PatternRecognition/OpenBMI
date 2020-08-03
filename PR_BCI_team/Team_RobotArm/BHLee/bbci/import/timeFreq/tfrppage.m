function [tfr,t,f] = tfrppage(x,t,N,h,trace);
%TFRPPAGE Pseudo Page time-frequency distribution.
%	[TFR,T,F]=TFRPPAGE(X,T,N,H,TRACE) computes the Pseudo Page 
%	distribution of a discrete-time signal X, or the
%	cross Pseudo Page representation between two signals. 
% 
%	X     : signal if auto-PPage, or [X1,X2] if cross-PPage.
%	T     : time instant(s)          (default : 1:length(X)).
%	N     : number of frequency bins (default : length(X)).
%	H     : frequency smoothing window, H(0) being forced to 1.
%	                                 (default : Hamming(N/4)). 
%	TRACE : if nonzero, the progression of the algorithm is shown
%	                                 (default : 0).
%	TFR   : time-frequency representation. When called without 
%	        output arguments, TFRPPAGE runs TFRQVIEW.
%	F     : vector of normalized frequencies.
%
%	Example :
%	 sig=fmlin(128,0.1,0.4); tfrppage(sig);
% 
%	See also all the time-frequency representations listed in
%	 the file CONTENTS (TFR*)

%	F. Auger, May-August 1994, July 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org

[xrow,xcol] = size(x);
if (nargin < 1),
 error('At least 1 parameter is required');
elseif nargin<=2,
 N=xrow;
end;

hlength=floor(N/4);
if (rem(hlength,2)==0),
 hlength=hlength+1;
end;

if (nargin == 1),
 t=1:xrow; h = window(hlength); trace=0;
elseif (nargin == 2 | nargin == 3),
 h = window(hlength); trace = 0;
elseif (nargin == 4),
 trace = 0;
end;

if (N<0),
 error('N must be greater than zero');
end;
[trow,tcol] = size(t);
if (xcol==0)|(xcol>2),
 error('X must have one or two columns');
elseif (trow~=1),
 error('T must only have one row'); 
elseif (2^nextpow2(N)~=N),
 fprintf('For a faster computation, N should be a power of two\n');
end; 

[hrow,hcol]=size(h); Lh=(hrow-1)/2; h=h/h(Lh+1);
if (hcol~=1)|(rem(hrow,2)==0),
 error('H must be a smoothing window with odd length');
end;

tfr= zeros (N,tcol) ;  
if trace, disp('Pseudo Page distribution'); end;
for icol=1:tcol,
 ti= t(icol); tau=max([-Lh,ti-N]):min([Lh,ti-1]);
 indices= rem(N+tau,N)+1;
 if trace, disprog(icol,tcol,10); end;
 tfr(indices,icol)=h(Lh+1+tau).*x(ti,1).*conj(x(ti-tau,xcol));
end; 
if trace, fprintf('\n'); end;
tfr= real(fft(tfr)); 

if (nargout==0),
 tfrqview(tfr,x,t,'tfrppage',h);
elseif (nargout==3),
 f=fftshift((-round(N/2):round(N/2)-1)/N)';
end;
