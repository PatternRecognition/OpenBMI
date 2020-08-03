%function ambifunt
%AMBIFUNT Unit test for the function ambifunb.

%	O. Lemoine - December 1995.
%	F. Auger - February 1996.

N=128; 

% Ambiguity function of a pulse
sig=((1:N)'==40); 
amb=ambifunb(sig);
[ik,jk]=find(amb~=0.0);
if any(jk~=N/2)|any(ik'-(1:N)),
 error('ambifunb test 1 failed');
end;

% Ambiguity function of a sine wave
sig=fmconst(N,0.2);
amb=ambifunb(sig);
[a b]=max(amb);
if any(b-odd(N/2)*ones(1,N-1)),
 error('ambifunb test 2 failed');
end;

% Energy 
sig=noisecg(N);
amb=ambifunb(sig);
if abs(abs(amb(odd(N/2),N/2))-norm(sig)^2)>sqrt(eps),
 error('ambifunb test 3 failed');
end;

% Link with the Wigner-Ville distribution
sig=fmlin(N);
amb=ambifunb(sig);
amb=amb([(N+rem(N,2))/2+1:N 1:(N+rem(N,2))/2],:);
ambi=ifft(amb).';
tdr=zeros(N); 		% Time-delay representation
tdr(1:N/2,:)=ambi(N/2:N-1,:);
tdr(N:-1:N/2+2,:)=ambi(N/2-1:-1:1,:);
wvd1=real(fft(tdr));
wvd2=tfrwv(sig);
errors=max(max(abs(wvd1-wvd2)));
if errors>sqrt(eps),
 error('ambifunb test 4 failed');
end;
				     

N=111;

% Ambiguity function of a pulse
sig=((1:N)'==40); 
amb=ambifunb(sig);
[ik,jk]=find(amb~=0.0);
if any(jk~=(N+1)/2)|any(ik'-(1:N)),
 error('ambifunb test 5 failed');
end;

% Ambiguity function of a sine wave
sig=fmconst(N,0.2);
amb=ambifunb(sig);
[a b]=max(amb);
if any(b-((N+1)/2)*ones(1,N)),
 error('ambifunb test 6 failed');
end;

% Energy 
sig=noisecg(N);
amb=ambifunb(sig);
if abs(abs(amb((N+1)/2,(N+1)/2))-norm(sig)^2)>sqrt(eps),
 error('ambifunb test 7 failed');
end;

% Link with the Wigner-Ville distribution
sig=fmlin(N);
amb=ambifunb(sig);
amb=amb([(N+1)/2:N 1:(N-1)/2],:);
ambi=ifft(amb).';
tdr=zeros(N); 		% Time-delay representation
tdr(1:(N+1)/2,:)      = ambi((N+1)/2:N,:);
tdr(N:-1:(N+1)/2+1,:) = ambi((N-1)/2:-1:1,:);
wvd1=real(fft(tdr));
wvd2=tfrwv(sig,1:N,N);
errors=max(max(abs(wvd1-wvd2)));
if errors>sqrt(eps),
 error('ambifunb test 8 failed');
end;
