%function fmconstt
%FMCONSTT Unit test for the function FMCONST.

%	F. Auger - December 1995, O. Lemoine - February 1996. 

 N=200;

 [sig iflaw]=fmconst(N);
 if any(abs(iflaw-0.25)>sqrt(eps))~=0 | abs(sig(N/2)-1)>sqrt(eps),
  error('fmconst test 1 failed');
 end;  

 [sig iflaw]=fmconst(N,0.1);
 if any(abs(iflaw-0.1)>sqrt(eps))~=0 | abs(sig(N/2)-1)>sqrt(eps),
  error('fmconst test 2 failed');
 end;

 [sig iflaw]=fmconst(N,0.1,20);
 if any(abs(iflaw-0.1)>sqrt(eps))~=0 | abs(sig(20)-1)>sqrt(eps), 
  error('fmconst test 3 failed');
 end;
 [ifl,t]=instfreq(sig);
 if norm(iflaw(t)-ifl)>sqrt(eps), 
  error('fmconst test 4 failed');
 end;


 N=211;

 [sig iflaw]=fmconst(N);
 if any(abs(iflaw-0.25)>sqrt(eps))~=0 | abs(sig(N/2))>sqrt(eps),
  error('fmconst test 5 failed');
 end;  

 [sig iflaw]=fmconst(N,0.1);
 if any(abs(iflaw-0.1)>sqrt(eps))~=0 | abs(sig(N/2)-1)>sqrt(eps),
  error('fmconst test 6 failed');
 end;

 [sig iflaw]=fmconst(N,0.1,20);
 if any(abs(iflaw-0.1)>sqrt(eps))~=0 | abs(sig(20)-1)>sqrt(eps), 
  error('fmconst test 7 failed');
 end;
 [ifl,t]=instfreq(sig);
 if norm(iflaw(t)-ifl)>sqrt(eps), 
  error('fmconst test 8 failed');
 end;
