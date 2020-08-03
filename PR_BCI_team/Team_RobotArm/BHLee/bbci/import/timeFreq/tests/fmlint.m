%function fmlint
%FMLINT	Unit test for the function FMLIN.

%	O. Lemoine - February 1996. 

 N=200;

 [sig iflaw]=fmlin(N);
 if abs(iflaw(1))>sqrt(eps) | abs(iflaw(N)-0.5)>sqrt(eps),
  error('fmlin test 1 failed');
 end;  

 [sig iflaw]=fmlin(N,0.1,0.4);
 if abs(iflaw(1)-0.1)>sqrt(eps) | abs(iflaw(N)-0.4)>sqrt(eps),
  error('fmlin test 2 failed');
 end;

 T0=111; 
 [sig iflaw]=fmlin(N,0.4,0.3,T0);
 if abs(sig(T0)-1)>sqrt(eps) | abs(iflaw(1)-0.4)>sqrt(eps) | ...
    abs(iflaw(N)-0.3)>sqrt(eps),  
  error('fmlin test 3 failed');
 end;
 [ifl,t]=instfreq(sig);
 if norm(iflaw(t)-ifl)>sqrt(eps), 
  error('fmlin test 4 failed');
 end;


 N=211;

 [sig iflaw]=fmlin(N);
 if abs(iflaw(1))>sqrt(eps) | abs(iflaw(N)-0.5)>sqrt(eps),
  error('fmlin test 5 failed');
 end;  

 [sig iflaw]=fmlin(N,0.1,0.4);
 if abs(iflaw(1)-0.1)>sqrt(eps) | abs(iflaw(N)-0.4)>sqrt(eps),
  error('fmlin test 6 failed');
 end;

 T0=111; 
 [sig iflaw]=fmlin(N,0.4,0.3,T0);
 if abs(sig(T0)-1)>sqrt(eps) | abs(iflaw(1)-0.4)>sqrt(eps) | ...
    abs(iflaw(N)-0.3)>sqrt(eps),  
  error('fmlin test 7 failed');
 end;
 [ifl,t]=instfreq(sig);
 if norm(iflaw(t)-ifl)>sqrt(eps), 
  error('fmlin test 8 failed');
 end;
