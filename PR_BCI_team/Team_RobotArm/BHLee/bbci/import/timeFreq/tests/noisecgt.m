%function noisecgt
%NOISECGT Unit test for the function NOISECG.

%	O. Lemoine - February 1996.

N=32768;
A1=0.2; A2=0.7;
sig=noisecg(N,A1,A2);


% Mean
Mean=mean(sig);
if abs(Mean)>10/sqrt(N),
 error('noisecg test 1 failed');
end


% Variance
Var=std(sig).^2;
if abs(Var-1)>10/sqrt(N),
 error('noisecg test 2 failed');
end


% histogram
Nh=100;
[h,m]=hist(real(sig),Nh); h=h/max(h);
Nc=find(abs(m)==min(abs(m)));
pdf=amgauss(Nh,Nc,Nh/(2*sqrt(pi)))';
if any(abs(h-pdf).^2>10/sqrt(N)),
 error('noisecg test 3 failed');
end


% whiteness
sig=noisecg(N);
autocor=xcorr(sig);
Max=max(autocor);
L=length(find(abs(autocor/Max)>5e-2));
if L/N>5e-4,
 error('noisecg test 4 failed');
end 

 
% For N=1 
N=1; Np=10000;
for k=1:Np,
 sig(k)=noisecg(N);
end
Mean=mean(sig);
if abs(Mean)>10/sqrt(Np),
 error('noisecg test 5 failed');
end
Var=std(sig).^2;
if abs(Var-1)>10/sqrt(Np),
 error('noisecg test 6 failed');
end
[h,m]=hist(real(sig),Nh); h=h/max(h);
Nc=find(abs(m)==min(abs(m)));
pdf=amgauss(Nh,Nc,Nh/(2*sqrt(pi)))';
if any(abs(h-pdf).^2>10/sqrt(Np)),
 error('noisecg test 7 failed');
end


% For N=2
N=2;
for k=1:2:(Np-1),
 noise=noisecg(N);
 sig(k)=noise(1);
 sig(k+1)=noise(2);
end
Mean=mean(sig);
if abs(Mean)>10/sqrt(Np),
 error('noisecg test 8 failed');
end
Var=std(sig).^2;
if abs(Var-1)>10/sqrt(Np),
 error('noisecg test 9 failed');
end
[h,m]=hist(real(sig),Nh); h=h/max(h);
Nc=find(abs(m)==min(abs(m)));
pdf=amgauss(Nh,Nc,Nh/(2*sqrt(pi)))';
if any(abs(h-pdf).^2>10/sqrt(Np)),
 error('noisecg test 10 failed');
end

