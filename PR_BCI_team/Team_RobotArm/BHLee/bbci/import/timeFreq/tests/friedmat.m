%function friedmat
%FRIEDMAT Unit test for the function FRIEDMAN.

%	O. Lemoine - April 1996.

N=128; 
t=1:N;

% The integral of a density is 1
sig=noisecg(N);
[tfr,rtfr,hat]=tfrrpwv(sig);
tifd=friedman(tfr,hat,t,'tfrrpwv'); 
unit=sum(sum(tifd));
if abs(unit-1)>sqrt(eps), 
 error('friedman test 1 failed');
end;


% Reality of the TIFD
if sum(any(abs(imag(tifd))>sqrt(eps)))~=0,
 error('friedman test 2 failed');
end


% Positivity
if any(any(tifd<0)),
 error('friedman test 3 failed');
end


% time localization
t0=60; sig=((1:N)'==t0);
[tfr,rtfr,hat]=tfrrppag(sig);
tifd=friedman(tfr,hat,t,'tfrrppag'); 
[ik,jk]=find(tifd~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('friedman test 4 failed');
end;


% frequency localization
f0=30;
sig=fmconst(N+6,f0/128);
[tfr,rtfr,hat]=tfrrsp(sig,N/2+2,N,window(129,'rect'));
tifd=friedman(tfr,hat,t,'tfrrsp'); 
if any(find(tifd>max(tifd)/N)~=f0+1),
 error('friedman test 5 failed');
end;


N=111; 
t=1:N;

% The integral of a density is 1
sig=noisecg(N);
[tfr,rtfr,hat]=tfrrpwv(sig);
tifd=friedman(tfr,hat,t,'tfrrpwv'); 
unit=sum(sum(tifd));
if abs(unit-1)>sqrt(eps), 
 error('friedman test 6 failed');
end;


% Reality of the TIFD
if sum(any(abs(imag(tifd))>sqrt(eps)))~=0,
 error('friedman test 7 failed');
end


% Positivity
if any(any(tifd<0)),
 error('friedman test 8 failed');
end


% time localization
t0=51; sig=((1:N)'==t0);
[tfr,rtfr,hat]=tfrrppag(sig);
tifd=friedman(tfr,hat,t,'tfrrppag'); 
[ik,jk]=find(tifd~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('friedman test 9 failed');
end;


% frequency localization
f0=31;
sig=fmconst(N+6,f0/N);
[tfr,rtfr,hat]=tfrrsp(sig,round(N/2)+2,N,window(N,'rect'));
tifd=friedman(tfr,hat,t,'tfrrsp'); 
if any(find(tifd>max(tifd)/N)~=f0+1),
 error('friedman test 10 failed');
end;

