function [tfr,t,f]=tfrideal(iflaws,t,N,trace)
%TFRIDEAL Ideal TFR for given instantaneous frequency laws.
%	[TFR,T,F]=TFRIDEAL(IFLAWS,T,N,TRACE) generates the ideal
%	time-frequency representation corresponding to the
%	instantaneous frequency laws of the components of a signal. 
%
%	IFLAWS : (M,P)-matrix where each column corresponds to
%		 the instantaneous frequency law of an (M,1)-signal,
%		 These P signals do not need to be present at the same time.
%		 The values of IFLAWS must be between 0 and 0.5.
%	T      : the time instant(s)      (default : 1:M).
%	N      : number of frequency bins (default : M).
%	TRACE  : if nonzero, the progression of the algorithm is shown
%                                         (default : 0).
%	TFR    : output time-frequency matrix, of size (N,length(t)).
%		 If nargout=0, a contour plot of TFR is automatically
%		 displayed on the screen.
%	F      : vector of normalized frequencies.
%
%	Example :
%         N=140; t=0:N-1; [x1,if1]=fmlin(N,0.05,0.3); 
%         [x2,if2]=fmsin(70,0.35,0.45,60);
%         if2=[zeros(35,1)*NaN;if2;zeros(35,1)*NaN];
%         tfrideal([if1 if2]); 
%
%	See also PLOTIFL, PLOTSID, and all the time-frequency 
%        representations listed in the CONTENTS file (TFR*)

%	O. Lemoine, F. Auger - March, April 1996.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin==0),
 error('at least one parameter required');
end;

[ifrow,ifcol]=size(iflaws);

if (nargin==1),
 t=1:ifrow; N=ifrow; trace=0;
elseif (nargin==2),
 N=ifrow; trace=0;
elseif (nargin==3),
 trace=0;
end;

[trow,tcol]=size(t);
if (trow~=1),
 error('T must only have one row'); 
end;

tfr=zeros(N,tcol);

if any(any(iflaws>0.5)) | any(any(iflaws<0)),
 error('The values of IFLAWS must be between 0 and 0.5');
end

if trace, disp('Ideal time-frequency distribution'); end;

for icol=1:tcol,
 if trace, disprog(icol,tcol,10); end;
 ti= t(icol); 
 for fi=1:ifcol,
  if isnan(iflaws(ti,fi)),
   tfr(fi,icol)=NaN;
  else
   tfr(round(iflaws(ti,fi)*2*(N-1))+1,icol)=1;
  end
 end
end;

if (nargout==0),
 f=(0:N-1)/(2*N); 
 contour(t,f,tfr,1,'y');
 xlabel('Time'); ylabel('Normalized frequency');
 title('Ideal time-frequency representation');
 grid;
elseif (nargout==3),
 f=(0.5*(0:N-1)/N)';
end;

