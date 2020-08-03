%function tfrspwvt
%TFRSPWVT Unit test for the function TFRSPWV.

%       F. Auger, Dec. 1995 - O. Lemoine, March 1996.

% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspwv(sig1);  
tfr2=tfrspwv(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrspwv test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrspwv(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrspwv test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrspwv(sig,1:N,N,[1]);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspwv test 3 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrspwv(sig,1:N,N,[1]);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrspwv test 4 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrspwv(sig,N/2+2,N,window(11,'rect'),window(N+1,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>2.0*eps),
 error('tfrspwv test 5 failed');
end;


% A SPWVD with a Dirac time-smoothing window is a PWVD
sig=noisecg(N);
tfr1=tfrspwv(sig,1:N,N,1);
tfr2=tfrpwv(sig);
if any(any(abs(tfr1-tfr2)>sqrt(eps))),
 error('tfrspwv test 6 failed');
end;


N=127;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspwv(sig1);  
tfr2=tfrspwv(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrspwv test 7 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrspwv(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrspwv test 8 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrspwv(sig,1:N,N,[1]);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspwv test 9 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrspwv(sig,1:N,N,[1]);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrspwv test 10 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrspwv(sig,round(N/2)+2,N,window(11,'rect'),window(N,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>sqrt(eps)),
 error('tfrspwv test 11 failed');
end;


% A SPWVD with a Dirac time-smoothing window is a PWVD
sig=noisecg(N);
tfr1=tfrspwv(sig,1:N,N,1);
tfr2=tfrpwv(sig);
if any(any(abs(tfr1-tfr2)>sqrt(eps))),
 error('tfrspwv test 12 failed');
end;
