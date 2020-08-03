%function tfruntet
%TFRUNTET Unit test for the time-frequency representation TFRUNTER.

%	O. Lemoine - June 1996. 

% We test each property of the corresponding TFR :

N=64;

% Active Unterberger distribution

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrunter(sig1,1:N,'A',0.02,0.48,440);  
tfr2=tfrunter(sig2,1:N,'A',0.02,0.48,440);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrunter test 1 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrunter(sig,1:N,'A',0.05,0.45,154);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrunter test 2 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrunter(sig,1:N,'A',0.1,0.4,N);
Es=norm(sig)^2;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrunter test 3 failed');
end

% Frequency-marginal
sig1=amgauss(N).*fmconst(N,.2);
sig2=amgauss(N).*fmconst(N,.3);
sig=sig1+sig2;
fmin=0.05; fmax=0.45; Nf=3*N;
[tfr,t,f]=tfrunter(sig,1:N,'A',fmin,fmax,Nf);
FFT=fft(sig);
nu=(0:N/2-1)/N;
psd1=abs(FFT(1:N/2)).^2;
psd2=integ(tfr,t)'/N/2;
psd1i=interp1(nu,psd1,f)/Nf/2;
if any(abs(psd1i-psd2')>1e-1),
 error('tfrunter test 4 failed');
end



% Passive Unterberger distribution

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrunter(sig1,1:N,'P',0.02,0.48,440);  
tfr2=tfrunter(sig2,1:N,'P',0.02,0.48,440);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrunter test 5 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrunter(sig,1:N,'P',0.05,0.45,154);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrunter test 6 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrunter(sig,1:N,'P',0.1,0.4,N);
Es=norm(sig)^2;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrunter test 7 failed');
end

% Time-marginal
sig=(amgauss(N,N/3)+amgauss(N,2*N/3)).*fmconst(N,.25);
fmin=0.05; fmax=0.45; Nf=3*N;
[tfr,t,f]=tfrunter(sig,1:N,'P',fmin,fmax,Nf);
ip1=abs(sig).^2/Nf;
ip2=integ(tfr',f')/N;
if any(abs(ip1(2:N)-ip2(1:N-1))>1e-3),
 error('tfrunter test 8 failed');
end

% Frequency-marginal
sig1=amgauss(N).*fmconst(N,.2);
sig2=amgauss(N).*fmconst(N,.3);
sig=sig1+sig2;
fmin=0.05; fmax=0.45; Nf=3*N;
[tfr,t,f]=tfrunter(sig,1:N,'P',fmin,fmax,Nf);
FFT=fft(sig);
nu=(0:N/2-1)/N;
psd1=abs(FFT(1:N/2)).^2;
psd2=integ(tfr,t)'/N/2;
psd1i=interp1(nu,psd1,f)/Nf/2;
if any(abs(psd1i-psd2')>1e-1),
 error('tfrunter test 9 failed');
end


N=63;

% Active Unterberger distribution

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrunter(sig,1:N,'A',0.05,0.45,155);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrunter test 10 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrunter(sig,1:N,'A',0.1,0.4,N);
SP = fft(hilbert(sig)); 
indmin = 1+round(.1*(N-2));
indmax = 1+round(.4*(N-2));
SPana = SP(indmin:indmax);
Es=SPana'*SPana/N;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrunter test 11 failed');
end

% Frequency-marginal
sig1=amgauss(N).*fmconst(N,.2);
sig2=amgauss(N).*fmconst(N,.3);
sig=sig1+sig2;
fmin=0.05; fmax=0.45; Nf=3*N;
[tfr,t,f]=tfrunter(sig,1:N,'A',fmin,fmax,Nf);
FFT=fft(sig);
nu=(0:round(N/2)-1)/N;
psd1=abs(FFT(1:round(N/2))).^2;
psd2=integ(tfr,t)'/N/2;
psd1i=interp1(nu,psd1,f)/Nf/2;
if any(abs(psd1i-psd2')>1e-1),
 error('tfrunter test 12 failed');
end



% Passive Unterberger distribution

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrunter(sig,1:N,'P',0.05,0.45,155);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrunter test 13 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrunter(sig,1:N,'P',0.1,0.4,N);
SP = fft(hilbert(sig)); 
indmin = 1+round(.1*(N-2));
indmax = 1+round(.4*(N-2));
SPana = SP(indmin:indmax);
Es=SPana'*SPana/N;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrunter test 14 failed');
end

% Time-marginal
sig=(amgauss(N,N/3)+amgauss(N,2*N/3)).*fmconst(N,.25);
fmin=0.05; fmax=0.45; Nf=3*N;
[tfr,t,f]=tfrunter(sig,1:N,'P',fmin,fmax,Nf);
ip1=abs(sig).^2/Nf;
ip2=integ(tfr',f')/N;
if any(abs(ip1(2:N)-ip2(1:N-1))>1e-3),
 error('tfrunter test 15 failed');
end

% Frequency-marginal
sig1=amgauss(N).*fmconst(N,.2);
sig2=amgauss(N).*fmconst(N,.3);
sig=sig1+sig2;
fmin=0.05; fmax=0.45; Nf=3*N;
[tfr,t,f]=tfrunter(sig,1:N,'P',fmin,fmax,Nf);
FFT=fft(sig);
nu=(0:round(N/2)-1)/N;
psd1=abs(FFT(1:round(N/2))).^2;
psd2=integ(tfr,t)'/N/2;
psd1i=interp1(nu,psd1,f)/Nf/2;
if any(abs(psd1i-psd2')>1e-1),
 error('tfrunter test 16 failed');
end

