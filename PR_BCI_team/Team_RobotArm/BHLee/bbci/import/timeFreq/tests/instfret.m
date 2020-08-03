%function instfret
%INSTFRET Unit test for the function INSTFREQ.

%	O. Lemoine - May 1996.

N=256; 

% Constant frequency modulation
[sig,ifl]=fmconst(N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 1 failed');
end;

[sig,ifl]=fmconst(N,0.01);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 2 failed');
end;


% Linear frequency modulation
[sig,ifl]=fmlin(N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 3 failed');
end;

[sig,ifl]=fmlin(N,0.02,0.36);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 4 failed');
end;


% Sinusoidal frequency modulation
[sig,ifl]=fmsin(N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>1e-4),
 error('instfreq test 5 failed');
end;

[sig,ifl]=fmsin(N,0.03,0.35,2*N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>1e-5),
 error('instfreq test 6 failed');
end;


N=221; 

% Constant frequency modulation
[sig,ifl]=fmconst(N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 7 failed');
end;

[sig,ifl]=fmconst(N,0.01);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 8 failed');
end;


% Linear frequency modulation
[sig,ifl]=fmlin(N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 9 failed');
end;

[sig,ifl]=fmlin(N,0.02,0.36);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>sqrt(eps)),
 error('instfreq test 10 failed');
end;


% Sinusoidal frequency modulation
[sig,ifl]=fmsin(N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>1e-4),
 error('instfreq test 11 failed');
end;

[sig,ifl]=fmsin(N,0.03,0.35,2*N);
[iflaw,t]=instfreq(sig);
if any(abs(ifl(t)-iflaw)>1e-5),
 error('instfreq test 12 failed');
end;

