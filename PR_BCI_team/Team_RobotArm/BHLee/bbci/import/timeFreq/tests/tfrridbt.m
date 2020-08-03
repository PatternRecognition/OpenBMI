%function tfrridbt
%TFRRIDBT Unit test for the function TFRRIDB.

%       F. Auger, Dec. 1995 - O. Lemoine, March 1996.

% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrridb(sig1);  
tfr2=tfrridb(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrridb test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrridb(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrridb test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrridb(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrridb test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrridb(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrridb test 4 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrridb(sig,1:N,N,[1],ones(N-1,1));
FFT=fft(sig);
psd1=abs(FFT(2:N/2)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:N/2))>sqrt(eps)),
 error('tfrridb test 5 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);noisecg(N/2);zeros(N/4,1)];
tfr=tfrridb(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrridb test 6 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrridb(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrridb test 7 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrridb(sig,N/2+2,N,window(11,'rect'),window(N+1,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>3.0*eps),
 error('tfrridb test 8 failed');
end;


N=123;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrridb(sig1);  
tfr2=tfrridb(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrridb test 9 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrridb(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrridb test 10 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrridb(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrridb test 11 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrridb(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrridb test 12 failed');
end


% Frequency-marginal
sig=noisecg(N);
tfr=tfrridb(sig,1:N,N,[1],ones(N,1));
FFT=fft(sig);
psd1=abs(FFT(2:fix(N/2)-3)).^2/(2*N);
psd2=mean(tfr(1:2:N,:)')';
if any(abs(psd1-psd2(2:fix(N/2)-3))>5e-2),
 error('tfrridb test 13 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);noisecg(round(N/2));zeros(round(N/4),1)];
tfr=tfrridb(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+2):N))>sqrt(eps))),
 error('tfrridb test 14 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrridb(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrridb test 15 failed');
end;


% frequency localization
f0=10;
sig=fmconst(N+6,f0/N);
tfr=tfrridb(sig,round(N/2)+2,N,window(11,'rect'),window(N,'rect'));
if (find(tfr>1/N)~=2*f0+1)|(abs(mean(tfr)-1.0)>sqrt(eps)),
 error('tfrridb test 16 failed');
end;
