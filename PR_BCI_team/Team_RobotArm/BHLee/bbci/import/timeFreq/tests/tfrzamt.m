%function tfrzamt
%TFRZAMT Unit test for the function TFRZAM.

%       O. Lemoine, March 1996.


% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrzam(sig1);  
tfr2=tfrzam(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrzam test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrzam(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrzam test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrzam(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrzam test 3 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);fmlin(N/2);zeros(N/4,1)];
tfr=tfrzam(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrzam test 4 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrzam(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrzam test 5 failed');
end;


N=127;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrzam(sig1);  
tfr2=tfrzam(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrzam test 6 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrzam(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrzam test 7 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrzam(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrzam test 8 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);fmlin(round(N/2));zeros(round(N/4),1)];
tfr=tfrzam(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+2):N))>sqrt(eps))),
 error('tfrzam test 9 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrzam(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrzam test 10 failed');
end;

