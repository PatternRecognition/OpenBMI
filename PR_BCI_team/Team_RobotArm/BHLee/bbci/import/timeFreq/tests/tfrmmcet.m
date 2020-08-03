function tfrmmcet
%TFRMMCET Unit test for the time frequency representation TFRMMCE.

%       O. Lemoine - March 1996. 

% We test each property of the corresponding TFR :

N=128;
h=zeros(19,3);
h(10+(-5:5),1)=window(11); 
h(10+(-7:7),2)=window(15);  
h(10+(-9:9),3)=window(19);

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrmmce(sig1,h);  
tfr2=tfrmmce(sig2,h);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrmmce test 1 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrmmce(sig,h);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrmmce test 2 failed');
end


% Positivity
if any(any(tfr<0)),
 error('tfrmmce test 3 failed');
end



N=121;
h=zeros(19,3);
h(10+(-6:6),1)=window(13,'gauss'); 
h(10+(-8:8),2)=window(17,'kaiser');  
h(10+(-7:7),3)=window(15,'parzen');

% Covariance by translation in time 
t1=55; t2=70; f=0.3;
sig1=amgauss(N,t1).*fmconst(N,f,t1); 
sig2=amgauss(N,t2).*fmconst(N,f,t2); 
tfr1=tfrmmce(sig1,h);  
tfr2=tfrmmce(sig2,h);        
[tr,tc]=size(tfr1);
nu=round(f*(tc-1)*2)+1;
tfr=tfr1-tfr2(:,modulo((1:tc)-t1+t2,tc));
if any(any(abs(tfr)>sqrt(eps))),
 error('tfrmmce test 4 failed');
end


% Reality of the TFR
sig=noisecg(N);
tfr=tfrmmce(sig,h);
if sum(any(abs(imag(tfr))>sqrt(eps)))~=0,
 error('tfrmmce test 5 failed');
end


% Positivity
if any(any(tfr<0)),
 error('tfrmmce test 6 failed');
end


