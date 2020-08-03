%function tfrmhst
%TFRMHST Unit test for the time frequency representation TFRMHS.

%       O. Lemoine - March 1996. 

% We test each property of the corresponding TFR :

N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrmhs(sig1);  
tfr2=tfrmhs(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrmhs test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrmhs(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrmhs test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrmhs(sig,1:N,N,[1],ones(N-1,1));
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrmhs test 3 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrmhs(sig,1:N,N,[1],ones(N-1,1));
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrmhs test 4 failed');
end;

		    

N=123;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrmhs(sig1);  
tfr2=tfrmhs(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrmhs test 5 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrmhs(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrmhs test 6 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrmhs(sig,1:N,N,[1],ones(N,1));
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrmhs test 7 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrmhs(sig,1:N,N,[1],ones(N,1));
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrmhs test 8 failed');
end;

		    
