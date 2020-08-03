%function tfrpaget
%TFRPAGET Unit test for the time frequency representation TFRPAGE.

%       O. Lemoine - March 1996. 

% We test each property of the corresponding TFR :


N=128;

% Covariance by translation in time 
t1=60; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrpage(sig1);  
tfr2=tfrpage(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrpage test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrpage(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrpage test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrpage(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrpage test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrpage(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrpage test 4 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrpage(sig);
FFT=fft(sig);
psd1=abs(FFT).^2/N;
psd2=mean(tfr')';
if any(abs(psd1-psd2)>sqrt(eps)),
 error('tfrpage test 5 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);noisecg(N/2);zeros(N/4,1)];
tfr=tfrpage(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrpage test 6 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrpage(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrpage test 7 failed');
end;


N=131;

% Covariance by translation in time 
t1=60; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrpage(sig1);  
tfr2=tfrpage(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrpage test 8 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrpage(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrpage test 9 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrpage(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrpage test 10 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrpage(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrpage test 11 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrpage(sig);
FFT=fft(sig);
psd1=abs(FFT).^2/N;
psd2=mean(tfr')';
if any(abs(psd1-psd2)>sqrt(eps)),
 error('tfrpage test 12 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);noisecg(round(N/2));zeros(round(N/4),1)];
tfr=tfrpage(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+2):N))>sqrt(eps))),
 error('tfrpage test 13 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrpage(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrpage test 14 failed');
end;

