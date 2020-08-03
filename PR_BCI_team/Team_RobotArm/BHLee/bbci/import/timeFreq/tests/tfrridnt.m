%function tfrridnt
%TFRRIDNT Unit test for the function TFRRIDBN.

%       O. Lemoine, June 1996.

% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrridbn(sig1);  
tfr2=tfrridbn(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrridbn test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrridbn(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrridbn test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrridbn(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrridbn test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrridbn(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrridbn test 4 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrridbn(sig,1:N,N,[1],ones(N-1,1));
FFT=fft(sig);
psd1=abs(FFT(2:N/2)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:N/2))>sqrt(eps)),
 error('tfrridbn test 5 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);noisecg(N/2);zeros(N/4,1)];
tfr=tfrridbn(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrridbn test 6 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrridbn(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrridbn test 7 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrridbn(sig,N/2+2,N,window(11,'rect'),window(N+1,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>3.0*eps),
 error('tfrridbn test 8 failed');
end;


clear; N=125;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrridbn(sig1);  
tfr2=tfrridbn(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrridbn test 9 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrridbn(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrridbn test 10 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrridbn(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrridbn test 11 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrridbn(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrridbn test 12 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrridbn(sig,1:N,N,[1],ones(N,1));
FFT=fft(sig);
psd1=abs(FFT(2:fix(N/2)-3)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:fix(N/2)-3))>5e-2),
 error('tfrridbn test 13 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);noisecg(round(N/2));zeros(round(N/4),1)];
tfr=tfrridbn(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+2):N))>sqrt(eps))),
 error('tfrridbn test 14 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrridbn(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrridbn test 15 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrridbn(sig,N/2+2,N,window(11,'rect'),window(N,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>sqrt(eps)),
 error('tfrridbn test 16 failed');
end;
