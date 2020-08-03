%function tfrrmsct
%TFRRMSCT Unit test for the function TFRRMSC.

%	O. Lemoine - April 1996. 

N=64;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrmsc(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrmsc test 1 failed');
end


% Energy conservation
%sig=fmlin(N);
%[tfr,rtfr]=tfrrmsc(sig);
%Es=norm(sig)^2;
%Etfr=sum(mean(rtfr));
%if abs(Es-Etfr)>sqrt(eps),
% error('tfrrmsc test 2 failed');
%end


% Positivity
if any(any(rtfr<0)),
 error('tfrrsp test 3 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrmsc(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(2:N)),
 error('tfrrmsc test 4 failed');
end;


% frequency localization
%f0=10;
%sig=fmconst(N+6,f0/N);
%[tfr rtfr]=tfrrmsc(sig,N/2+2,N);
%if any(find(rtfr>max(rtfr)/N)~=f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
% error('tfrrmsc test 5 failed');
%end;



N=65;

% Reality of the TFR
sig=noisecg(N);
[tfr,rtfr]=tfrrmsc(sig);
if sum(any(abs(imag(rtfr))>sqrt(eps)))~=0,
 error('tfrrmsc test 6 failed');
end


% Energy conservation
%sig=fmlin(N);
%[tfr,rtfr]=tfrrmsc(sig);
%Es=norm(sig)^2;
%Etfr=sum(mean(rtfr));
%if abs(Es-Etfr)>sqrt(eps),
% error('tfrrmsc test 7 failed');
%end


% Positivity
if any(any(rtfr<0)),
 error('tfrrsp test 8 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
[tfr,rtfr]=tfrrmsc(sig);
[ik,jk]=find(abs(rtfr)>sqrt(eps));
if any(jk~=t0)|any(ik'-(2:N)),
 error('tfrrmsc test 9 failed');
end;


% frequency localization
%f0=10;
%sig=fmconst(N+6,f0/N);
%[tfr rtfr]=tfrrmsc(sig,round(N/2)+2,N);
%if any(find(rtfr>max(rtfr)/N)~=f0+1)|(abs(mean(rtfr)-1.0)>sqrt(eps)),
% error('tfrrmsc test 10 failed');
%end;


