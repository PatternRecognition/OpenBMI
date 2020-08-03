function disprog(i,N,steps);
%DISPROG Display progression of a loop.
%	DISPROG(i,N,steps) displays the progression of a loop.
%
%	I     : loop variable
%	N     : final value of i
%	STEPS : number of displayed steps.
%
%	Example:
%	 N=16; for i=1:N, disprog(i,N,5); end;

%	F. Auger, August, December 1995.
%       from an idea of R. Settineri.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

global begin_time_disprog ;

if (i==1),
 begin_time_disprog=cputime;
end;

if (i==N),
 fprintf('100 %% complete in %g seconds.\n', cputime-begin_time_disprog);
 clear begin_time_disprog;
elseif (floor(i*steps/N)~=floor((i-1)*steps/N)),
 fprintf('%g ', floor(i*steps/N)*ceil(100/steps));
end;


