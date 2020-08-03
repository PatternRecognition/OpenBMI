function [sig,locatoms]= atoms(N,coord);
%ATOMS	Linear combination of elementary Gaussian atoms.
%	[SIG,LOCATOMS] = ATOMS(N,COORD) 
%	generates a signal consisting in a linear combination of elementary
%	gaussian wave packets. The locations of the time-frequency centers 
% 	of the different atoms are either fixed by the input parameter COORD 
%	or successively defined by clicking with the mouse (if NARGIN==1).  
% 
%	N        : number of points of the signal
%	COORD    : matrix of time-frequency centers, of the form
%		   [t1,f1,T1,A1;...;tM,fM,TM,AM]. (ti,fi) are the 
%		   time-frequency coordinates of atom i, Ti is its time 
%		   duration and Ai its amplitude. Frequencies f1..fM should 
%		   be normalized (between 0 and 0.5). 
%		   If nargin==1, the location of the atoms will be defined
%		   by clicking with the mouse, with the help of a menu. The 
%		   default value for Ti is N/4.
%	SIG      : output signal.
%	LOCATOMS : matrix of time-frequency coordinates and durations of the
%		   atoms. 
%
%	Example : 
%	 sig=atoms(128);
%	 sig=atoms(128,[32,0.3,32,1;56,0.15,48,1.22;102,0.41,20,0.7]); 

%	P. Flandrin, May 1995 - O. Lemoine, February 1996.
%	F. Auger - O. Lemoine, June 1996.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%			    lemoine@alto.unice.fr 

if ( nargin < 1 ),
 error ( 'At least one parameter required' ) ;
end

sig=(1+j)*zeros(N,1);
locatoms=[];
t=linspace(0,2*pi,100); 
Natoms=0; choice=1;
T=N/4; A=1;

clf; 
axes('position',[0.10 0.12 0.80 0.45]);
plot([]); zoom off; grid on
axis([1 N 0 0.5]); hold on;
xlabel('Time'); 
ylabel('Normalized frequency');

if (nargin==1),
 fprintf(' Default value for the time-duration : %f',T);
 fprintf('\n Default value for the amplitude     : %f',A);fprintf('\n');
 while choice~=5,
  choice=menu('ATOMS MENU',...
	'Add a gaussian atom',...
	'Delete the last atom',...
	'Change the time-duration',...
	'Change the amplitude',...
	'Stop');
  if choice==1, 
   [t0,f0]=ginput(1); 
   t0=round(max(min(t0,N),1)); f0=max(min(f0,0.5),0.0);
   locatoms=[locatoms;t0 f0 T A];
   plot(t0,f0,'x');
   plot((t0+j*f0)+(0.5*T*cos(t)+j*(2/(T*pi))*sin(t)))
   sig=sig + A*amgauss(N,t0,T) .* fmconst(N,f0,t0);
   Natoms=Natoms+1;
  elseif (choice==2 & Natoms>=1),
   t0=locatoms(Natoms,1);
   f0=locatoms(Natoms,2);
   T =locatoms(Natoms,3);
   A =locatoms(Natoms,4);
   sig=sig - A*amgauss(N,t0,T) .* fmconst(N,f0,t0);
   plot(t0,f0,'kx');
   plot((t0+j*f0)+(0.5*T*cos(t)+j*(2/(T*pi))*sin(t)),'k')
   locatoms(Natoms,:)=[];
   Natoms=Natoms-1;
  elseif choice==3,
   fprintf(' Old time duration : %f', T);
   T=input(' New time duration : ');
  elseif choice==4,
   fprintf(' Old amplitude : %f', A);
   A=input(' New amplitude : ');
  end
 end;
elseif (nargin==2),
 [Natoms,ccoord]=size(coord);
 if (ccoord~=4),
   error('Bad dimension for COORD');
 end
 for k=1:Natoms,
  t0=round(max(min(coord(k,1),N),1));
  f0=max(min(coord(k,2),0.5),0.0);
  T=coord(k,3); A=coord(k,4);
  if t0~=coord(k,1),
   disp('Warning : ti should be between 1 and N');
  end
  if f0~=coord(k,2),
   disp('Warning : fi should be between 0 and 0.5');
  end
  if T<0,
   error('T must be positive');
  end
  if A<0,
   error('A must be positive');
  end
  sig=sig+A*amgauss(N,t0,T) .* fmconst(N,f0,t0); 
  plot(t0,f0,'x');
  plot((t0+j*f0)+(0.5*T*cos(t)+j*(2/(T*pi))*sin(t)))
 end
 locatoms=coord;
end

hold off
axes('position',[0.10,0.65 0.80 0.25]);
Min=min(real(sig));
Max=max(real(sig));
if Max<sqrt(eps),
 sig=zeros(N,1);
end
plot(1:N, real(sig),'g');
if Min<Max,
 axis([1 N Min Max]);
else
 axis([1 N -1 1]);
end
title([int2str(Natoms),' Gaussian atom(s)'])
grid on
