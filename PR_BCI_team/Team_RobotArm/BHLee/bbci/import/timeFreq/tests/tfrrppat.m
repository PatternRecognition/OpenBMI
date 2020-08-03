%function tfrrppat
%TFRRPPAT Unit test for the function TFRRPPAG.

%	O. Lemoine - April 1996. 


N=128;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrppag(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrppag test 1 failed');
end


% Energy conservation
sig=fmlin(N);
[tfr,rtfr]=tfrrppag(sig);
Es=norm(sig)^2;
Etfr=sum(mean(rtfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrrppag test 2 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrppag(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrrppag test 3 failed');
end;


% frequency localization
f0=30;
sig=fmconst(N+6,f0/N);
[tfr rtfr]=tfrrppag(sig,N/2+2,N,window(N+1,'rect'));
if any(find(rtfr>2*max(rtfr)/N)~=f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
 error('tfrrppag test 4 failed');
end;


N=131;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrppag(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrppag test 5 failed');
end


% Energy conservation
sig=fmlin(N);
[tfr,rtfr]=tfrrppag(sig);
Es=norm(sig)^2;
Etfr=sum(mean(rtfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrrppag test 6 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrppag(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrrppag test 7 failed');
end;


% frequency localization
f0=30;
sig=fmconst(N+6,f0/N);
[tfr rtfr]=tfrrppag(sig,round(N/2)+2,N,window(N,'rect'));
if any(find(rtfr>2*max(rtfr)/N)~=f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
 error('tfrrppag test 8 failed');
end;

