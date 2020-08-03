function [tfr,t,f] = tfrpage(x,t,N,trace);
%TFRPAGE Page time-frequency distribution.
%	[TFR,T,F]=TFRPAGE(X,T,N,TRACE) computes the Page distribution
%	of a discrete-time signal X, 
%	or the cross Page representation between two signals. 
% 
%	X     : signal if auto-Page, or [X1,X2] if cross-Page.
%	T     : time instant(s).
%	N     : number of frequency bins  (default : length(X)).
%	TRACE : if nonzero, the progression of the algorithm is shown
%	                                  (default : 0).
%	TFR   : time-frequency representation. When called without 
%               output arguments, TFRPAGE runs TFRQVIEW.
%	F     : vector of normalized frequencies.
%
%	Example :
%	 sig=fmlin(128,0.1,0.4); tfrpage(sig);
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
elseif (nargin == 1),
 t=1:xrow; N=xrow; trace=0;
elseif (nargin == 2),
 N=xrow; trace=0;
elseif (nargin == 3),
 trace = 0;
end;

if (N<0),
 error('N must be greater than zero');
end;
[trow,tcol] = size(t);
if (xcol==0) | (xcol>2),
 error('X must have one or two columns');
elseif (trow~=1),
 error('T must only have one row'); 
elseif (rem(log(N)/log(2),1)~=0),
 fprintf('For a faster computation, N should be a power of two\n');
end; 

tfr= zeros (N,tcol) ;  
if trace, disp('Page distribution'); end;
for icol=1:tcol,
 ti=t(icol); tau=-min([N-ti,xrow-ti]):(ti-1);
 indices=rem(N+tau,N)+1;
 if trace, disprog(icol,tcol,10); end;
 tfr(indices,icol)= x(ti,1)*conj(x(ti-tau,xcol));
end; 
if trace, fprintf('\n'); end;
tfr= real(fft(tfr)); 

if (nargout==0),
 tfrqview(tfr,x,t,'tfrpage');
elseif (nargout==3),
 f=fftshift((-N/2:N/2-1)/N)';
end;


