%function tfrspt
%TFRSPT Unit test for the function TFRSP.

%       F. Auger, Dec. 1995 - O. Lemoine, March 1996.

% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=60; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrsp(sig1);  
tfr2=tfrsp(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrsp test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrsp(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrsp test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrsp(sig,1:N,N,[1]);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrsp test 3 failed');
end


% Positivity
if any(any(tfr<0)),
 error('tfrsp test 4 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrsp(sig,1:N,N,[1]);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrsp test 5 failed');
end;


N=131;

% Covariance by translation in time 
t1=61; t2=73; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrsp(sig1);  
tfr2=tfrsp(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrsp test 6 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrsp(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrsp test 7 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrsp(sig,1:N,N,[1]);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrsp test 8 failed');
end


% Positivity
if any(any(tfr<0)),
 error('tfrsp test 9 failed');
end


% time localization
t0=31; sig=((1:N)'==t0);
tfr=tfrsp(sig,1:N,N,[1]);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrsp test 10 failed');
end;
