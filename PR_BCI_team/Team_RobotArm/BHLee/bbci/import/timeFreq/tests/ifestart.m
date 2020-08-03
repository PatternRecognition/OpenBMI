%function ifestart
%IFESTART Unit test for the function IFESTAR2.

%	O. Lemoine - August 1996.

N=235; 

% Constant frequency modulation
[sig,ifl]=fmconst(N);
[iflaw,t]=ifestar2(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('ifestar2 test 1 failed');
end;

[sig,ifl]=fmconst(N,0.01);
[iflaw,t]=ifestar2(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('ifestar2 test 2 failed');
end;


% Linear frequency modulation
[sig,ifl]=fmlin(N,.05,.45);
[iflaw,t]=ifestar2(sig);
if any(abs(ifl(t)-iflaw)>1e-2),
 error('ifestar2 test 3 failed');
end;


% Sinusoidal frequency modulation
[sig,ifl]=fmsin(N);
[iflaw,t]=ifestar2(sig);
if any(abs(ifl(t)-iflaw)>1e-1),
 error('ifestar2 test 4 failed');
end;

[sig,ifl]=fmsin(N,0.03,0.35,2*N);
[iflaw,t]=ifestar2(sig);
if any(abs(ifl(t)-iflaw)>1e-2),
 error('ifestar2 test 5 failed');
end;

