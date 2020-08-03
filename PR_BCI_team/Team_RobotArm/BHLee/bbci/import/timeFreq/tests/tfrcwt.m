%function tfrcwt
%TFRCWT Unit test for the function TFRCW.

%       F. Auger, Dec. 1995 - O. Lemoine, March 1996. 

% We test each property of the corresponding TFR :

N=128; 

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrcw(sig1);  
tfr2=tfrcw(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrcw test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrcw(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrcw test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrcw(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrcw test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrcw(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrcw test 4 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrcw(sig,1:N,N,[1],ones(N-1,1));
FFT=fft(sig);
psd1=abs(FFT(2:N/2)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:N/2))>sqrt(eps)),
 error('tfrcw test 5 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrcw(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrcw test 6 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrcw(sig,N/2+2,N,window(11,'rect'),window(N+1,'rect'),1.2);
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>2.0*eps),
 error('tfrcw test 7 failed');
end;



N=127; 

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrcw(sig1);  
tfr2=tfrcw(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrcw test 8 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrcw(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrcw test 9 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrcw(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrcw test 10 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrcw(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrcw test 11 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrcw(sig,1:N,N,[1],ones(N,1));
FFT=fft(sig);
psd1=abs(FFT(2:fix(N/2)-2)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:fix(N/2)-2))>5e-2),
 error('tfrcw test 12 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrcw(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrcw test 13 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrcw(sig,round(N/2)+2,N,window(11,'rect'),window(N,'rect'),1.2);
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>sqrt(eps)),
 error('tfrcw test 14 failed');
end;
