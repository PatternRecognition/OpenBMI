%function ridgest
%RIDGEST Unit test for the function RIDGES.

%	O. Lemoine - April 1996.

N=128; 
t=1:N;

% time localization
t0=60; sig=((1:N)'==t0);
[tfr,rtfr,hat]=tfrrsp(sig);
[ptt,ptf]=ridges(tfr,hat,t,'tfrrsp'); 
if any(ptt~=t0)|any(abs(ptf'-(1:N)/N)>sqrt(eps)),
 error('ridges test 1 failed');
end;


% frequency localization
f0=30;
sig=fmconst(N,f0/128);
[tfr,rtfr,hat]=tfrrsp(sig);
[ptt,ptf]=ridges(tfr,hat,t,'tfrrsp'); 
if any((ptf(3:106)-(f0+1)/N)>sqrt(eps)),
 error('ridges test 2 failed');
end;
if any(abs(ptt(3:106)-(13:116)')>sqrt(eps)),
 error('ridges test 3 failed');
end;


N=117; 
t=1:N;

% time localization
t0=53; sig=((1:N)'==t0);
[tfr,rtfr,hat]=tfrrsp(sig);
[ptt,ptf]=ridges(tfr,hat,t,'tfrrsp'); 
if any(ptt~=t0)|any(abs(ptf'-(1:N)/N)>sqrt(eps)),
 error('ridges test 4 failed');
end;


% frequency localization
f0=31;
sig=fmconst(N,f0/N);
[tfr,rtfr,hat]=tfrrsp(sig);
[ptt,ptf]=ridges(tfr,hat,t,'tfrrsp'); 
if any((ptf(3:95)-(f0+1)/N)>sqrt(eps)),
 error('ridges test 5 failed');
end;
if any(abs(ptt(3:95)-(11:103)')>sqrt(eps)),
 error('ridges test 6 failed');
end;

