function [tm,D2]=momftfr(tfr,tmin,tmax,time);
%MOMFTFR Frequency moments of a time-frequency representation.
%	[TM,D2]=MOMFTFR(TFR,TMIN,TMAX,TIME) computes the frequeny 
%	moments of a time-frequency representation.
% 
%	TFR    : time-frequency representation ([Nrow,Ncol]size(TFR)). 
%	TMIN   : smallest column element of TFR taken into account
%	                            (default : 1) 
%	TMAX   : highest column element of TFR taken into account
%	                            (default : Ncol)
%	TIME   : true time instants (default : 1:Ncol)
%	TM     : averaged time          (first order moment)
%	D2     : squared time duration  (second order moment)
%
%	Example :
%	 sig=fmlin(128,0.1,0.4); 
%	 [tfr,t,f]=tfrwv(sig); [tm,D2]=momftfr(tfr); 
%	 subplot(211); plot(f,tm); subplot(212); plot(f,D2);
%
%	See also MOMTTFR, MARGTFR.

%	F. Auger, August 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

[tfrrow,tfrcol]=size(tfr);
if (nargin==1),
 tmin=1; tmax=tfrcol; time=tmin:tmax;
elseif (nargin==2),
 tmax=tfrcol; time=tmin:tmax;
elseif (nargin==3),
 time=tmin:tmax;
end;

if (tmin>tmax)|(tmin<=0)|(tmax>tfrcol),
 error('1<=TMIN<=TMAX<=Ncol');
end;

E  = sum(tfr(:,tmin:tmax).');
tm = (time    * tfr(:,tmin:tmax).' ./E).'; 
D2 = (time.^2 * tfr(:,tmin:tmax).' ./E).' - tm.^2;


