%function tfrpwvt
%TFRPWVT Unit test for the function TFRPWV.

%       F. Auger, Dec. 1995 - O. Lemoine, March 1996.

% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrpwv(sig1);  
tfr2=tfrpwv(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrpwv test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrpwv(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrpwv test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrpwv(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrpwv test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrpwv(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrpwv test 4 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);noisecg(N/2);zeros(N/4,1)];
tfr=tfrpwv(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrpwv test 5 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrpwv(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrpwv test 6 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrpwv(sig,N/2+2,N,window(N+1,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>2.0*eps),
 error('tfrpwv test 7 failed');
end;


% A PWVD with a large rectangular window is a WVD
sig=noisecg(N);
tfr1=tfrpwv(sig,1:N,N,ones(2*N+1,1));
tfr2=tfrwv(sig);
if any(any(abs(tfr1-tfr2)>1e-5)),
 error('tfrpwv test 8 failed');
end;


N=127;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrpwv(sig1);  
tfr2=tfrpwv(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrpwv test 9 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrpwv(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrpwv test 10 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrpwv(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrpwv test 11 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrpwv(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrpwv test 12 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);noisecg(round(N/2));zeros(round(N/4),1)];
tfr=tfrpwv(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+2):N))>sqrt(eps))),
 error('tfrpwv test 13 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrpwv(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrpwv test 14 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrpwv(sig,round(N/2)+2,N,window(N,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>sqrt(eps)),
 error('tfrpwv test 15 failed');
end;


% A PWVD with a large rectangular window is a WVD
sig=noisecg(N);
tfr1=tfrpwv(sig,1:N,N,ones(2*N+1,1));
tfr2=tfrwv(sig);
if any(any(abs(tfr1-tfr2)>1e-5)),
 error('tfrpwv test 16 failed');
end;
