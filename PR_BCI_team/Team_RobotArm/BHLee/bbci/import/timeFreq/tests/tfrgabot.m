%function tfrgabot
%TFRGABT Unit test for the function tfrgabor.

%	O. Lemoine - February 1996.

N1=256; 
M=16;
N=16;

% Test of the biorthogonality between h and gam (dual frame window) 
Nh=65;					% length of window h
h=amgauss(Nh); h=h/norm(h);		% h must be of odd length
n0=1;
m0=1;
sig=zeros(N1,1);
sig(((n0-1)*N+1):((n0-1)*N+Nh))=h.*fmconst(Nh,(m0-1)/M);		
tfr=tfrgabor(sig,N,1,h);		% Critical sampling case
if abs(tfr(1,1)-1)>sqrt(eps),
  error('tfrgabor test 1 failed');
end
errors=find(any(tfr>eps));
if length(errors)~=1,
  error('tfrgabor test 2 failed');
end

% Localization
Nh=37;					% length of window h
h=amgauss(Nh); h=h/norm(h);		% h must be of odd length
n0=8;
m0=8;
sig=zeros(N1,1);
sig(((n0-1)*N+1):((n0-1)*N+Nh))=h.*fmconst(Nh,(m0-1)/M);		
tfr=tfrgabor(sig,N,1,h);		% Critical sampling case
if abs(tfr(m0,n0)-1)>sqrt(eps),		% C(n0,m0)=1
  error('tfrgabor test 3 failed');
end
errors=find(any(tfr>eps));		% Only one non-zero coeff
if length(errors)~=1,
  error('tfrgabor test 4 failed');
end

% For another window and position of sig
Nh=19;					% length of window h
h=amexpo2s(Nh);	h=h/norm(h);		% h must be of odd length
n0=13;
m0=5;
sig=zeros(N1,1);
sig(((n0-1)*N+1):((n0-1)*N+Nh))=h.*fmconst(Nh,(m0-1)/M);		
tfr=tfrgabor(sig,N,1,h);		% Critical sampling case
if abs(tfr(m0,n0)-1)>sqrt(eps),		% C(n0,m0)=1
  error('tfrgabor test 5 failed');
end
errors=find(any(tfr>eps));		% Only one non-zero coeff
if length(errors)~=1,
  error('tfrgabor test 6 failed');
end

% For another window and position of sig
Nh=17;					% length of window h
h=amexpo2s(Nh);	h=h/norm(h);		% h must be of odd length
n0=15;
m0=8;
sig=zeros(N1,1);
sig(((n0-1)*N+1):((n0-1)*N+Nh))=h.*fmconst(Nh,(m0-1)/M);		
tfr=tfrgabor(sig,N,1,h);		% Critical sampling case
if abs(tfr(m0,n0)-1)>sqrt(eps),		% C(n0,m0)=1
  error('tfrgabor test 7 failed');
end
errors=find(any(tfr>eps));		% Only one non-zero coeff
if length(errors)~=1,
  error('tfrgabor test 8 failed');
end


% Synthesis for q=1
Nh=33;q=1;
h=amgauss(Nh); h=h/norm(h);
sig=zeros(N1,1);
sig=amgauss(N1);
sigr=zeros(N1,1);
[tfr,dgr]=tfrgabor(sig,N,q,h);
alpha=round((2*N1/N-1-Nh)/(2*q));
hN1=zeros(N1,1); 
hN1((N1-(Nh-1))/2-alpha:(N1+Nh-1)/2-alpha)=h;	
n=1:N;
som=M*ifft(dgr);
for k=1:N1,
  indice=modulo(k-round(n*M/q),N1);
  sigr(k)=som(modulo(k,M),:)*fftshift(hN1(indice));
end
if any(abs(sig-sigr)>sqrt(eps))~=0,	
  error('tfrgabor test 9 failed');
end

% Synthesis for q=4
Nh=33;q=4;N=32;M=32;
h=amgauss(Nh); h=h/norm(h);
sig=zeros(N1,1);
sig=amgauss(N1);
sigr=zeros(N1,1);
[tfr,dgr]=tfrgabor(sig,N,q,h);
alpha=round((2*N1/N-1-Nh)/(2*q));
hN1=zeros(N1,1); 
hN1((N1-(Nh-1))/2-alpha:(N1+Nh-1)/2-alpha)=h;	
n=1:N;
som=M*ifft(dgr);
for k=1:N1,
  indice=modulo(k-round(n*M/q),N1);
  sigr(k)=som(modulo(k,M),:)*fftshift(hN1(indice));
end
if any(abs(sig-sigr)>sqrt(eps))~=0,	
  error('tfrgabor test 10 failed');
end



