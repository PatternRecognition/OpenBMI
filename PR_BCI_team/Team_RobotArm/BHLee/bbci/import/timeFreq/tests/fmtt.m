%function fmtt
%FMTT 	Unit test for the function FMT.

%	O. Lemoine - May 1996.

N=128; 

% Perfect reconstruction with ifmt
fmin=0.1; fmax=0.5;
sig=amgauss(N).*fmconst(N,.3); 
[MELLIN,BETA]=fmt(sig,fmin,fmax,N);
X=ifmt(MELLIN,BETA,N); 
err=abs(X-sig);
if any(err>1e-7),
 error('fmt test 1 failed');
end


% Energy conservation
x=fmconst(N);
fmin=0.1; fmax=0.4;
FMT=fmt(x,fmin,fmax,N);
SP=fft(x); 
indmin = 1+round(fmin*(N-2));
indmax = 1+round(fmax*(N-2));
SPana=SP(indmin:indmax);
nu=(indmin:indmax)'/N; 
SPp=SPana./nu;
Es=SPp'*SPana;
Efmt=norm(FMT)^2;
if abs(Es-Efmt)>sqrt(eps),
 error('fmt test 2 failed');
end;


% Unitarity of the MT
x1=amgauss(N).*fmlin(N,.15,.35);
x2=amgauss(N).*fmconst(N);
fmin=0.01; fmax=0.49;
FMT1=fmt(x1,fmin,fmax,2*N);
FMT2=fmt(x2,fmin,fmax,2*N);
indmin = 1+round(fmin*(2*N-2));
indmax = 1+round(fmax*(2*N-2));
SP1=fft(x1); SP2=fft(x2);
nu=(indmin:indmax)'/N/2; 
SP1p=SP1(indmin:indmax)./nu;
cor1=SP1p'*SP2(indmin:indmax);
cor2=conj(FMT1*FMT2');
if abs(cor1-cor2)>N*1e-2,
 error('fmt test 3 failed');
end;


% Covariance by dilation 
% Property of the Mellin transform used in scale.
% So as scale works, this property is verified.


% MT of a product = convolution of the MT 
x1=amgauss(N).*fmlin(N,.15,.35);
x2=amgauss(N).*fmsin(N,.15,.35);
FMT1=fmt(x1,fmin,fmax,2*N);
FMT2=fmt(x2,fmin,fmax,2*N);
FMT=conv(FMT1,FMT2);
FMT=FMT/max(real(FMT));
X1=fft(x1); X2=fft(x2);
X=X1.*X2; 
x=fftshift(ifft(X)); 
FMTp=fmt(x,fmin,fmax,2*N);
FMTp=FMTp/max(real(FMTp));
Diff=FMTp-FMT(N+1:3*N);		     
if any(abs(Diff)>1e-4),
 error('fmt test 4 failed');
end



N=121; 

% Perfect reconstruction with ifmt
fmin=0.1; fmax=0.5;
sig=amgauss(N).*fmconst(N,.3); 
[MELLIN,BETA]=fmt(sig,fmin,fmax,N+1);
X=ifmt(MELLIN,BETA,N); 
err=abs(X-sig);
if any(err>1e-2),
 error('fmt test 5 failed');
end


% Energy conservation
x=fmconst(N);
fmin=0.1; fmax=0.4;
FMT=fmt(x,fmin,fmax,N+1);
SP=fft(hilbert(real(x))); 
indmin = 1+round(fmin*(N-2));
indmax = 1+round(fmax*(N-2));
SPana=SP(indmin:indmax);
nu=(indmin:indmax)'/(N+1); 
SPp=SPana./nu;
Es=SPp'*SPana;
Efmt=norm(FMT)^2;
if abs(Es-Efmt)>sqrt(eps),
 error('fmt test 6 failed');
end;


% Unitarity of the MT
x1=amgauss(N).*fmlin(N,.15,.35);
x2=amgauss(N).*fmconst(N);
fmin=0.01; fmax=0.49;
FMT1=fmt(x1,fmin,fmax,2*N);
FMT2=fmt(x2,fmin,fmax,2*N);
indmin = 1+round(fmin*(2*N-2));
indmax = 1+round(fmax*(2*N-2));
SP1=fft(x1); SP2=fft(x2);
nu=(indmin:indmax)'/N/2; 
SP1p=SP1(indmin:indmax)./nu;
cor1=SP1p'*SP2(indmin:indmax);
cor2=conj(FMT1*FMT2');
if abs(cor1-cor2)>N*1e-2,
 error('fmt test 7 failed');
end;


% MT of a product = convolution of the MT 
x1=amgauss(N).*fmlin(N,.15,.35);
x2=amgauss(N).*fmsin(N,.15,.35);
FMT1=fmt(x1,fmin,fmax,2*N);
FMT2=fmt(x2,fmin,fmax,2*N);
FMT=conv(FMT1,FMT2);
FMT=FMT/max(real(FMT));
X1=fft(x1); X2=fft(x2);
X=X1.*X2; 
x=fftshift(ifft(X)); 
FMTp=fmt(x,fmin,fmax,2*N);
FMTp=FMTp/max(real(FMTp));
Diff=FMTp-FMT(N+1:3*N);		     
if any(abs(Diff)>1e-4),
 error('fmt test 8 failed');
end

