%function tfrppagt
%TFRPPAGT Unit test for the time frequency representation TFRPPAGE.

%       O. Lemoine - March 1996. 

% We test each property of the corresponding TFR :


N=128;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrppage(sig1);  
tfr2=tfrppage(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrppage test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrppage(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrppage test 2 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrppage(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrppage test 3 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrppage(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrppage test 4 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(N/4,1);noisecg(N/2);zeros(N/4,1)];
tfr=tfrppage(sig);
if sum(any(abs(tfr(:,1:N/4-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(3*N/4+1):N))>sqrt(eps))),
 error('tfrppage test 5 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrppage(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrppage test 6 failed');
end;


% A PPAGED with a constant window is a PAGED
sig=noisecg(N);
tfr1=tfrppage(sig,1:N,N,ones(2*N+1,1));
tfr2=tfrpage(sig);
if any(any(abs(tfr1-tfr2)>sqrt(eps))),
 error('tfrppage test 7 failed');
end


N=129;

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrppage(sig1);  
tfr2=tfrppage(sig2);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrppage test 8 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrppage(sig);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrppage test 9 failed');
end


% Energy conservation
sig=noisecg(N);
tfr=tfrppage(sig);
Es=norm(sig)^2;
Etfr=sum(mean(tfr));
if abs(Es-Etfr)>sqrt(eps),
 error('tfrppage test 10 failed');
end


% Time-marginal
sig=noisecg(N);
tfr=tfrppage(sig);
ip1=abs(sig).^2;
ip2=mean(tfr)';
if any(abs(ip1-ip2)>sqrt(eps)),
 error('tfrppage test 11 failed');
end


% Conservation of the time support (wide-sense)
sig=[zeros(round(N/4),1);noisecg(round(N/2));zeros(round(N/4),1)];
tfr=tfrppage(sig);
if sum(any(abs(tfr(:,1:round(N/4)-1))>sqrt(eps))) | ...
   sum(any(abs(tfr(:,(round(3*N/4)+1):N))>sqrt(eps))),
 error('tfrppage test 12 failed');
end


% time localization
t0=30; sig=((1:N)'==t0);
tfr=tfrppage(sig);
[ik,jk]=find(tfr~=0.0);
if any(jk~=t0)|any(ik'-(1:N)),
 error('tfrppage test 13 failed');
end;


% A PPAGED with a constant window is a PAGED
sig=noisecg(N);
tfr1=tfrppage(sig,1:N,N,ones(2*N+1,1));
tfr2=tfrpage(sig);
if any(any(abs(tfr1-tfr2)>sqrt(eps))),
 error('tfrppage test 14 failed');
end

