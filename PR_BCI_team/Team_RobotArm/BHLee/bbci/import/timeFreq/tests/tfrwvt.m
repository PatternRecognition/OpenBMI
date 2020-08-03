%function tfrwvt
%TFRWVT	Unit test for the function TFRWV.

%	F. Auger, December 1995 - O. Lemoine, March 1996.

N=128; 

% Covariance by translation in time 
t1=60; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrwv(sig1);  
tfr2=tfrwv(sig2);        
[tr,tc]=size(tfr1);
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrwv test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrwv(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrwv test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrwv(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrwv test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrwv(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrwv test 4 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrwv(sig);
FFT=fft(sig);
psd1=abs(FFT(2:N/2)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:N/2))>sqrt(eps)),
 error('tfrwv test 5 failed');
end


% Compatibility with filtering
h=amgauss(N,N/2,sqrt(N)).*fmlin(N);
x=amgauss(N).*fmconst(N);
y=conv(x,h);  y=y((N/2+1):3*N/2);
tfrx=tfrwv(x);
tfry=tfrwv(y)/(2*N);
tfrh=tfrwv(h);
tfr=zeros(N);
for f=1:N,
 tmp=conv(tfrx(f,:),tfrh(f,:));
 tfr(f,:)=tmp((N/2+1):3*N/2)/N;
end
if any(any(abs(tfr-tfry)>sqrt(eps))),
 error('tfrwv test 6 failed');
end


% Unitarity
x1=amgauss(N).*fmlin(N);
x2=amgauss(N).*fmsin(N);
tfr1=tfrwv(x1);
tfr2=tfrwv(x2);
cor1=abs(x1'*x2)^2/2;
cor2=sum(sum(tfr1.*conj(tfr2)))/N;
if abs(cor1-cor2)>sqrt(eps),
 error('tfrwv test 7 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);fmlin(N/2);zeros(N/4,1)];
tfr=tfrwv(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrwv test 8 failed');
end


% time localization
t0=30; sig= ((1:N)'==t0);
tfr=tfrwv(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrwv test 9 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrwv(sig,N/2+2,N);
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>2.0*eps),
 error('tfrwv test 10 failed');
end;


N=127; 

% Covariance by translation in time 
t1=60; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrwv(sig1);  
tfr2=tfrwv(sig2);        
[tr,tc]=size(tfr1);
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrwv test 11 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrwv(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrwv test 12 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrwv(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrwv test 13 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrwv(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrwv test 14 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrwv(sig);
FFT=fft(sig);
psd1=abs(FFT(2:fix(N/2)-3)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:fix(N/2)-3))>5e-2),
 error('tfrwv test 15 failed');
end


% Compatibility with filtering
h=amgauss(N,N/2,sqrt(N)).*fmlin(N);
x=amgauss(N).*fmconst(N);
y=conv(x,h);  y=y((round(N/2)+1):round(3*N/2));
tfrx=tfrwv(x);
tfry=tfrwv(y)/(2*N);
tfrh=tfrwv(h);
tfr=zeros(N);
for f=1:N,
 tmp=conv(tfrx(f,:),tfrh(f,:));
 tfr(f,:)=tmp((round(N/2)+1):round(3*N/2))/N;
end
if any(any(abs(tfr-tfry)>sqrt(eps))),
 error('tfrwv test 16 failed');
end


% Unitarity
x1=amgauss(N).*fmlin(N);
x2=amgauss(N).*fmsin(N);
tfr1=tfrwv(x1);
tfr2=tfrwv(x2);
cor1=abs(x1'*x2)^2/2;
cor2=sum(sum(tfr1.*conj(tfr2)))/N;
if abs(cor1-cor2)>sqrt(eps),
 error('tfrwv test 17 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);fmlin(round(N/2));zeros(round(N/4),1)];
tfr=tfrwv(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+2):N))>sqrt(eps))),
 error('tfrwv test 18 failed');
end


% time localization
t0=30; sig= ((1:N)'==t0);
tfr=tfrwv(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrwv test 19 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrwv(sig,round(N/2)+2,N);
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>sqrt(eps)),
 error('tfrwv test 20 failed');
end;
