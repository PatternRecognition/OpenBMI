%function tfrrit
%TFRRIT Unit test for the function TFRRI.

%       O. Lemoine, March 1996.

N=128; 

% Covariance by translation in time 
t1=60; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrri(sig1);  
tfr2=tfrri(sig2);        
[tr,tc]=size(tfr1);
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrri test 1 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrri(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrri test 2 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrri(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrri test 3 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrri(sig);
FFT=fft(sig);
psd1=abs(FFT(2:N/2)).^2/N;
psd2=real(mean(tfr(2:N/2,:)'))';
if any(abs(psd1-psd2)>sqrt(eps)),
 error('tfrri test 4 failed');
end


% Unitarity
x1=amgauss(N).*fmlin(N);
x2=amexpo2s(N);
tfr1=tfrri(x1);
tfr2=tfrri(x2);
cor1=abs(x1'*x2)^2;
cor2=real(sum(sum(tfr1.*conj(tfr2)))/N);
if abs(cor1-cor2)>sqrt(eps),
 error('tfrri test 5 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);fmlin(N/2);zeros(N/4,1)];
tfr=tfrri(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrri test 6 failed');
end


% time localization
t0=30; sig= ((1:N)'==t0);
tfr=tfrri(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrri test 7 failed');
end;


N=127; 

% Covariance by translation in time 
t1=61; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrri(sig1);  
tfr2=tfrri(sig2);        
[tr,tc]=size(tfr1);
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>1e-7)),
 error('tfrri test 8 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrri(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrri test 9 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrri(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrri test 10 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrri(sig);
FFT=fft(sig);
psd1=abs(FFT(2:fix(N/2))).^2/N;
psd2=real(mean(tfr(2:fix(N/2),:)'))';
if any(abs(psd1-psd2)>sqrt(eps)),
 error('tfrri test 11 failed');
end


% Unitarity
x1=amgauss(N).*fmlin(N);
x2=amexpo2s(N);
tfr1=tfrri(x1);
tfr2=tfrri(x2);
cor1=abs(x1'*x2)^2;
cor2=real(sum(sum(tfr1.*conj(tfr2)))/N);
if abs(cor1-cor2)>sqrt(eps),
 error('tfrri test 12 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);fmlin(round(N/2));zeros(round(N/4),1)];
tfr=tfrri(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+2):N))>sqrt(eps))),
 error('tfrri test 13 failed');
end


% time localization
t0=30; sig= ((1:N)'==t0);
tfr=tfrri(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrri test 14 failed');
end;

