function plotifl(t,iflaws);
%PLOTIFL Plot normalized instantaneous frequency laws.
% 	PLOTIFL(T,IFLAWS) plot the normalized instantaneous frequency 
% 	laws of each signal component.
%
% 	T      : time instants,
% 	IFLAWS : (M,P)-matrix where each column corresponds to
%		 the instantaneous frequency law of an (M,1)-signal,
%		 These P signals do not need to be present at the same 
%		 time instants.
%		 The values of IFLAWS must be between -0.5 and 0.5.
%
% 	Example : 
%        N=140; t=0:N-1; [x1,if1]=fmlin(N,0.05,0.3); 
%        [x2,if2]=fmsin(70,0.35,0.45,60);
%        if2=[zeros(35,1)*NaN;if2;zeros(35,1)*NaN];
%        plotifl(t,[if1 if2]);
%
%	See also tfrideal, plotsid.

% 	F. Auger, August 94, August 95.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

if (nargin<2),
 error('2 parameters are required'); 
end;

[trow,tcol] = size(t);
[ifrow,ifcol]=size(iflaws); 

indices=find(1-isnan(iflaws));
maxif=max(max(iflaws(indices))); 
minif=min(min(iflaws(indices)));

if (trow~=1),
 error('t must only have one row'); 
end ;
if (maxif > 0.5) | (minif < -0.5),
 disp('Each element of IFLAWS should be between -0.5 and 0.5'); 
end;


if (minif>=0),
 plot([t(1),t(tcol)],[0.0 0.5],'i');
 hold on;
else
 plot([t(1),t(tcol)],[-0.5 0.5],'i');
 hold on; 
end;
plot(t,iflaws); hold off;

grid
xlabel('Time');
ylabel('Normalized frequency');
title('Instantaneous frequency law(s)');