%function ambifuwt
%AMBIFUWT Unit test for the function AMBIFUWB.

%	O. Lemoine - April 1996.

clear;
N=128;
t0=45;
t=1:N;

% Ambiguity function of a pulse : an hyperbola
sig=(t'==t0); 
[amb,tau,theta]=ambifuwb(sig,0.005,0.5,3*N);
for ti=1:82,
 [Max ind(ti)]=max(abs(amb(:,ti).^2)); 
end
hyp1=theta(ind);
taup=1:82;
hyp2=log(20.23./(-taup+85));
if any(abs(hyp1-hyp2)>.1),
 error('ambifuwb test 1 failed');
end;


% Ambiguity function of a sine wave : non zero only for a=1 ie theta=0.
f0=0.25;
sig=fmconst(N,f0);
[amb,tau,theta]=ambifuwb(sig,0.1,0.4,N);
for ti=1:112,
 [Max ind(ti)]=max(abs(amb(:,ti).^2)); 
end
if any(ind~=N/2+1),
 error('ambifuwb test 2 failed');
end;


% Energy 
sig=fmlin(N).*amgauss(N);
Es=norm(sig)^2; 
[amb,tau,theta]=ambifuwb(sig,0.05,0.45,N);
Eamb=abs(amb(N/2,N/2));
if abs(Eamb-Es)>sqrt(eps),
 error('ambifuwb test 3 failed');
end;


clear;
N=121;
t=1:N;
t0=45;
		     
% Ambiguity function of a pulse : an hyperbola
sig=(t'==t0); 
[amb,tau,theta]=ambifuwb(sig,0.005,0.5,3*N);
for ti=1:74,
 [Max ind(ti)]=max(abs(amb(:,ti).^2)); 
end
hyp1=theta(ind);
taup=1:74;
hyp2=log(17./(-taup+79));
if any(abs(hyp1-hyp2)>5e-2),
 error('ambifuwb test 4 failed');
end;


% Ambiguity function of a sine wave : non zero only for a=1 ie theta=0.
f0=0.25;
sig=fmconst(N,f0);
[amb,tau,theta]=ambifuwb(sig,0.1,0.4,N);
for ti=1:105,
 [Max ind(ti)]=max(abs(amb(:,ti).^2)); 
end
if any(ind~=round(N/2)),
 error('ambifuwb test 5 failed');
end;


% Energy 
sig=fmlin(N).*amgauss(N);
Es=norm(sig)^2; 
[amb,tau,theta]=ambifuwb(sig,0.05,0.45,N);
Eamb=abs(amb(round(N/2),round(N/2)));
if abs(Eamb-Es)>sqrt(eps),
 error('ambifuwb test 6 failed');
end;


			     