%function tfrspawt
%TFRSPAWT Unit test for the time-frequency representation TFRSPAW.

%	O. Lemoine - June 1996. 

% We test each property of the corresponding TFR :

N=64;

K=-1;
NG0=15;
NH0=14;

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspaw(sig1,1:N,K,NH0,NG0,0.05,0.45,N);  
tfr2=tfrspaw(sig2,1:N,K,NH0,NG0,0.05,0.45,N);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrspaw test 1 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrspaw(sig,1:N,K,NH0,NG0,0.05,0.45,N);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrspaw test 2 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrspaw(sig,1:N,K,NH0,NG0,0.1,0.4,N);
Es=norm(sig)^2;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspaw test 3 failed');
end


K=0;
NG0=2;
NH0=11;

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspaw(sig1,1:N,K,NH0,NG0,0.05,0.45,N);  
tfr2=tfrspaw(sig2,1:N,K,NH0,NG0,0.05,0.45,N);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrspaw test 4 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrspaw(sig,1:N,K,NH0,NG0,0.05,0.45,N);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrspaw test 5 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrspaw(sig,1:N,K,NH0,NG0,0.1,0.4,N);
Es=norm(sig)^2;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspaw test 6 failed');
end


K=1/2;
NG0=5;
NH0=12;

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspaw(sig1,1:N,K,NH0,NG0,0.05,0.45,N);  
tfr2=tfrspaw(sig2,1:N,K,NH0,NG0,0.05,0.45,N);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrspaw test 7 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrspaw(sig,1:N,K,NH0,NG0,0.05,0.45,N);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrspaw test 8 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrspaw(sig,1:N,K,NH0,NG0,0.1,0.4,N);
Es=norm(sig)^2;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspaw test 9 failed');
end


K=2;
NG0=7;
NH0=17;

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspaw(sig1,1:N,K,NH0,NG0,0.05,0.45,2*N);  
tfr2=tfrspaw(sig2,1:N,K,NH0,NG0,0.05,0.45,2*N);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrspaw test 10 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrspaw(sig,1:N,K,NH0,NG0,0.05,0.45,2*N);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrspaw test 11 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrspaw(sig,1:N,K,NH0,NG0,0.1,0.4,N);
Es=norm(sig)^2;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspaw test 12 failed');
end



N=63;

K=-1;
NG0=15;
NH0=14;

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspaw(sig1,1:N,K,NH0,NG0,0.05,0.45,N);  
tfr2=tfrspaw(sig2,1:N,K,NH0,NG0,0.05,0.45,N);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrspaw test 13 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrspaw(sig,1:N,K,NH0,NG0,0.05,0.45,N);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrspaw test 14 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrspaw(sig,1:N,K,NH0,NG0,0.1,0.4,N);
SP = fft(hilbert(real(sig))); 
indmin = 1+round(.1*(N-2));
indmax = 1+round(.4*(N-2));
SPana = SP(indmin:indmax);
Es=SPana'*SPana/N;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspaw test 15 failed');
end


K=0;
NG0=2;
NH0=11;

% Covariance by translation in time 
t1=30; t2=40; f=0.25; 
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrspaw(sig1,1:N,K,NH0,NG0,0.05,0.45,N);  
tfr2=tfrspaw(sig2,1:N,K,NH0,NG0,0.05,0.45,N);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if max(max(abs(tfr)))>1e-1,
 error('tfrspaw test 15 failed');
end

% Reality of the TFR
sig=noisecg(N); 
tfr=tfrspaw(sig,1:N,K,NH0,NG0,0.05,0.45,N);
if any(any(abs(imag(tfr))>sqrt(eps))),
 error('tfrspaw test 16 failed');
end

% Energy conservation
sig=fmconst(N); 
[tfr,t,f]=tfrspaw(sig,1:N,K,NH0,NG0,0.1,0.4,N);
SP = fft(hilbert(real(sig))); 
indmin = 1+round(.1*(N-2));
indmax = 1+round(.4*(N-2));
SPana = SP(indmin:indmax);
Es=SPana'*SPana/N;
Etfr=integ2d(tfr,t,f);
if abs(Es-Etfr)>sqrt(eps),
 error('tfrspaw test 17 failed');
end

