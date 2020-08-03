function [N,M]=divider(N1);
%DIVIDER Find dividers of an integer.  
%	[N,M]=DIVIDER(N1) find two integers N and M such that M*N=N1 and
%	M and N as close as possible from sqrt(N1).
%
%	Example :
%	 N1=258; [N,M]=divider(N1)

%	F. Auger - November 1995.
%	Copyright (c) 1996 by CNRS (France).
%
%	------------------- CONFIDENTIAL PROGRAM -------------------- 
%	This program can not be used without the authorization of its
%	author(s). For any comment or bug report, please send e-mail to 
%	f.auger@ieee.org 

N=floor(sqrt(N1));
continue=1.0;
while continue,
 Nold=N;
 M=ceil(N1/N);
 N=floor(N1/M);
 continue=(N~=Nold);
end;
