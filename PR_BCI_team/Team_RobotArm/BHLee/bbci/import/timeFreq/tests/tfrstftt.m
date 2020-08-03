%function tfrstftt
%TFRSTFTT Unit test for the function TFRSTFT.

%	O. Lemoine - April 1996. 

N=128;

% Covariance by translation in time 
t1=60; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=abs(tfrstft(sig1)).^2;  
tfr2=abs(tfrstft(sig2)).^2;        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrstft test 1 failed');
end


% Unitarity
x1=amgauss(N).*fmlin(N);
x2=amexpo2s(N);
tfr1=tfrstft(x1);
tfr2=tfrstft(x2);
cor1=x1'*x2;
cor2=sum(sum(conj(tfr1).*tfr2))/N;
if abs(cor1-cor2)>sqrt(eps),
 error('tfrstft test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=abs(tfrstft(sig,1:N,N,[1])).^2;
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrstft test 3 failed');
end


% Comparison with the spectrogram
sig=amgauss(N).*fmlin(N);
t=1:N; Nf=64; Lh=8; 
h=window(2*Lh+1,'Kaiser'); h=h/norm(h);
tfr1=abs(tfrstft(sig,t,Nf,h)).^2;
tfr2=tfrsp(sig,t,Nf,h);
if any(any(abs(tfr1-tfr2)>sqrt(eps))),
 error('tfrstft test 4 failed');
end


% Synthesis
t=1:N; Lh=8; 
h=window(2*Lh+1,'Kaiser'); h=h/norm(h);
sig=fmlin(N,0.1,0.4); 
stft=tfrstft(sig,t,N,h); 
timerep=ifft(stft); sig2=zeros(N,1);
for ti=1:N,
  tau=-min([N/2-1,Lh,N-ti]):min([N/2-1,Lh,ti-1]);
  indices= rem(N+tau,N) + 1 + N*(ti-tau-1); 
  sig2(ti)=timerep(indices)*h(Lh+1+tau)/norm(h(Lh+1+tau));
end;
if any(abs(sig2(Lh+1:N-Lh)-sig(Lh+1:N-Lh))>sqrt(eps)),
 error('tfrstft test 5 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=abs(tfrstft(sig,1:N,N,[1])).^2;
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrstft test 6 failed');
end;


N=127;

% Covariance by translation in time 
t1=61; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=abs(tfrstft(sig1)).^2;  
tfr2=abs(tfrstft(sig2)).^2;        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrstft test 7 failed');
end


% Unitarity
x1=amgauss(N).*fmlin(N);
x2=amexpo2s(N);
tfr1=tfrstft(x1);
tfr2=tfrstft(x2);
cor1=x1'*x2;
cor2=sum(sum(conj(tfr1).*tfr2))/N;
if abs(cor1-cor2)>sqrt(eps),
 error('tfrstft test 8 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=abs(tfrstft(sig,1:N,N,[1])).^2;
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrstft test 9 failed');
end


% Comparison with the spectrogram
sig=amgauss(N).*fmlin(N);
t=1:N; Nf=64; Lh=8; 
h=window(2*Lh+1,'Kaiser'); h=h/norm(h);
tfr1=abs(tfrstft(sig,t,Nf,h)).^2;
tfr2=tfrsp(sig,t,Nf,h);
if any(any(abs(tfr1-tfr2)>sqrt(eps))),
 error('tfrstft test 10 failed');
end


% Synthesis
t=1:N; Lh=8; 
h=window(2*Lh+1,'Kaiser'); h=h/norm(h);
sig=fmlin(N,0.1,0.4); 
stft=tfrstft(sig,t,N,h); 
timerep=ifft(stft); sig2=zeros(N,1);
for ti=1:N,
  tau=-min([round(N/2)-1,Lh,N-ti]):min([round(N/2)-1,Lh,ti-1]);
  indices= rem(N+tau,N) + 1 + N*(ti-tau-1); 
  sig2(ti)=timerep(indices)*h(Lh+1+tau)/norm(h(Lh+1+tau));
end;
if any(abs(sig2(Lh+1:N-Lh)-sig(Lh+1:N-Lh))>sqrt(eps)),
 error('tfrstft test 11 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=abs(tfrstft(sig,1:N,N,[1])).^2;
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrstft test 12 failed');
end;

