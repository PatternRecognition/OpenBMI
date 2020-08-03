%function tfrbertt
%TFRBERTT Unit test for the time-frequency representation TFRBERT.

%	O. Lemoine - June 1996. 

% We test each property of the corresponding TFR :

N=64;

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrbert(sig1,1:N,0.01,0.49,468);  
tfr2=tfrbert(sig2,1:N,0.01,0.49,468);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrbert test 1 failed');
end


% Reality of the TFR
sig=noisecg(N); 
tfr=tfrbert(sig,1:N,0.05,0.45,2*N);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrbert test 2 failed');
end


% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrbert(sig,1:N,0.1,0.4,N);
Es=norm(sig)^2;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrbert test 3 failed');
end


% Frequency-marginal
sig1=amgauss(N).*fmconst(N,.2);
sig2=amgauss(N).*fmconst(N,.3);
sig=sig1+sig2;
fmin=0.05; fmax=0.45; Nf=2*N;
[tfr,t,f]=tfrbert(sig,1:N,fmin,fmax,Nf);
FFT=fft(sig);
nu=(0:N/2-1)/N;
psd1=abs(FFT(1:N/2)).^2;
psd2=integ(tfr,t)'/N/2;
psd1i=interp1(nu,psd1,f)/Nf/2;
if any(abs(psd1i-psd2')>1e-1),
 error('tfrbert test 4 failed');
end


N=61;

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrbert(sig,1:N,0.05,0.45,2*N+1);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrbert test 5 failed');
end


% Energy conservation
sig=fmconst(N); fmin=0.1; fmax=0.4;
[tfr,t,f]=tfrbert(sig,1:N,fmin,fmax,N);
SP = fft(hilbert(sig)); 
indmin = 1+round(fmin*(N-2));
indmax = 1+round(fmax*(N-2));
SPana = SP(indmin:indmax);
Es=SPana'*SPana/N;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrbert test 6 failed');
end


% Frequency-marginal
sig1=amgauss(N).*fmconst(N,.2);
sig2=amgauss(N).*fmconst(N,.3);
sig=sig1+sig2;
fmin=0.05; fmax=0.45; Nf=2*N+1;
[tfr,t,f]=tfrbert(sig,1:N,fmin,fmax,Nf);
FFT=fft(sig);
nu=(0:round(N/2)-1)/N;
psd1=abs(FFT(1:round(N/2))).^2;
psd2=integ(tfr,t)'/N/2;
psd1i=interp1(nu,psd1,f)/Nf/2;
if any(abs(psd1i-psd2')>1e-1),
 error('tfrbert test 7 failed');
end

