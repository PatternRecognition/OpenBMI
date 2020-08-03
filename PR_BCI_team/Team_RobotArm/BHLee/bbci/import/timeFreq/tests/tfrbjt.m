%function tfrbjt
%TFRBJT	Unit test for the function TFRBJ.

%	F. Auger, Dec. 1995 - O. Lemoine, March 1996.


% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrbj(sig1);  
tfr2=tfrbj(sig2);        
[tr,tc]=size(tfr1);
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrbj test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrbj(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrbj test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrbj(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrbj test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrbj(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrbj test 4 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrbj(sig,1:N,N,[1],ones(N-1,1));
FFT=fft(sig);
psd1=abs(FFT(2:N/2)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:N/2))>sqrt(eps)),
 error('tfrbj test 5 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);fmlin(N/2);zeros(N/4,1)];
tfr=tfrbj(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrbj test 6 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrbj(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrbj test 7 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrbj(sig,N/2+2,N,window(11,'rect'),window(N+1,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>2.0*eps),
 error('tfrbj test 8 failed');
end;
sig=fmconst(N,0.1);
tfr=tfrbj(sig,1:N,N/2,window(11,'rect'),window(N/2+1,'rect'));
if (sum(mean(tfr))~=sum(abs(sig).^2)),
 error('tfrbj test 9 failed');
end;


N=111;

% Reality of the TFR
sig=noisecg(N);
tfr=tfrbj(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrbj test 10 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrbj(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrbj test 11 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrbj(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrbj test 12 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrbj(sig,1:N,N,[1],ones(N,1));
FFT=fft(sig);
psd1=abs(FFT(2:round(N/2))).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1(3:round(N/2)-3)-psd2(4:round(N/2)-2))>5e-2),
 error('tfrbj test 13 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);fmlin(round(N/2));zeros(round(N/4),1)];
tfr=tfrbj(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*round(N/4)+1):N))>sqrt(eps))),
 error('tfrbj test 14 failed');
end


% time localization
t0=31; sig=((1:N)'==t0);
tfr=tfrbj(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrbj test 15 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrbj(sig,round(N/2)+2,N,window(11,'rect'),window(N,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>sqrt(eps)),
 error('tfrbj test 16 failed');
end;
sig=fmconst(N,0.1);
tfr=tfrbj(sig,1:N,round(N/2),window(11,'rect'),window(round(N/2)+1,'rect'));
if abs(sum(mean(tfr))-sum(abs(sig).^2))>sqrt(eps),
 error('tfrbj test 17 failed');
end;
