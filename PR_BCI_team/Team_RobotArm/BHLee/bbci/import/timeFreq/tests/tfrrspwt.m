%function tfrrspwt
%TFRRSPWT Unit test for the function TFRRSPWV.

%	F. Auger - December 1995, O. Lemoine - April 1996. 


N=128;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrspwv(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrspwv test 1 failed');
end


% Energy conservation
sig=fmlin(N);
[tfr,rtfr]=tfrrspwv(sig);
Es=norm(sig)^2;
Etfr=sum(mean(rtfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrrspwv test 2 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrspwv(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrrspwv test 3 failed');
end;


% frequency localization
f0=30;
sig=fmconst(N+6,f0/N);
[tfr rtfr]=tfrrspwv(sig,N/2+2,N,window(N+1,'rect'));
if any(find(rtfr>max(rtfr)/N)~=2*f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
 error('tfrrspwv test 4 failed');
end;


% A RSPWVD with a Dirac time-smoothing window is a RPWVD
sig=noisecg(N);
[tfr1 rtfr1]=tfrrspwv(sig,1:N,N/2,1);
[tfr2 rtfr2]=tfrrpwv(sig,1:N,N/2);
if any(any(abs(rtfr1-rtfr2)>sqrt(eps))),
 error('tfrrspwv test 5 failed');
end;


N=111;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrspwv(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrspwv test 6 failed');
end


% Energy conservation
sig=fmlin(N);
[tfr,rtfr]=tfrrspwv(sig);
Es=norm(sig)^2;
Etfr=sum(mean(rtfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrrspwv test 7 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrspwv(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrrspwv test 8 failed');
end;


% frequency localization
f0=30;
sig=fmconst(N+6,f0/N);
[tfr rtfr]=tfrrspwv(sig,round(N/2)+2,N,window(N,'rect'));
if any(find(rtfr>max(rtfr)/N)~=2*f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
 error('tfrrspwv test 9 failed');
end;


% A RSPWVD with a Dirac time-smoothing window is a RPWVD
sig=noisecg(N);
[tfr1 rtfr1]=tfrrspwv(sig,1:N,round(N/2),1);
[tfr2 rtfr2]=tfrrpwv(sig,1:N,round(N/2));
if any(any(abs(rtfr1-rtfr2)>sqrt(eps))),
 error('tfrrspwv test 10 failed');
end;
