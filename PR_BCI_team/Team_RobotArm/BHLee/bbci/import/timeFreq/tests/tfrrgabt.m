%function tfrrgabt
%TFRRGABT Unit test for the function TFRRGAB.

%	O. Lemoine - April 1996. 

N=128;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrgab(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrgab test 1 failed');
end


% Energy conservation
sig=fmlin(N);
[tfr,rtfr]=tfrrgab(sig);
Es=norm(sig)^2;
Etfr=sum(mean(rtfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrrgab test 2 failed');
end


% Positivity
if any(any(rtfr<0)),
 error('tfrrsp test 3 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrgab(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrrgab test 4 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
[tfr rtfr]=tfrrgab(sig,N/2+2,N,N+1);
if any(find(rtfr>max(rtfr)/N)~=f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
 error('tfrrgab test 5 failed');
end;



N=127;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrgab(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrgab test 6 failed');
end


% Energy conservation
sig=fmlin(N);
[tfr,rtfr]=tfrrgab(sig);
Es=norm(sig)^2;
Etfr=sum(mean(rtfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrrgab test 7 failed');
end


% Positivity
if any(any(rtfr<0)),
 error('tfrrsp test 8 failed');
end


% time localization
t0=31; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrgab(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrrgab test 9 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
[tfr rtfr]=tfrrgab(sig,round(N/2)+2,N,N);
if any(find(rtfr>max(rtfr)/N)~=f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
 error('tfrrgab test 10 failed');
end;


