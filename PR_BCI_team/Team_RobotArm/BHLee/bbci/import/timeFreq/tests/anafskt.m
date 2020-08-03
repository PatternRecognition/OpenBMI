%function anafskt
%ANAFSKT Unit test for the function ANAFSK.

%	O. Lemoine - February 1996.

N=256;

% Output frequency law
% At each frequency shift, on point of the iflaw is moved 
Nbf=10; Nc=32;
[signal,if]=anafsk(N,Nc,Nbf);
iflaw=instfreq(signal);
if sum(abs(iflaw-if(2:N-1))>sqrt(eps))>ceil((N-Nc)/Nc),
  error('anafsk test 1 failed');
end

Nbf=5; Nc=41;
[signal,if]=anafsk(N,Nc,Nbf);
iflaw=instfreq(signal);
if sum(abs(iflaw-if(2:N-1))>sqrt(eps))>ceil((N-Nc)/Nc),
  error('anafsk test 2 failed');
end

Nbf=15; Nc=57;
[signal,if]=anafsk(N,Nc,Nbf);
iflaw=instfreq(signal);
if sum(abs(iflaw-if(2:N-1))>sqrt(eps))>ceil((N-Nc)/Nc),
  error('anafsk test 3 failed');
end

% Output amplitude
Nbf=22; Nc=26;
[signal,if]=anafsk(N,Nc,Nbf);
if any(abs(abs(signal)-1)>sqrt(eps))~=0,
  error('anafsk test 4 failed');
end

Nbf=3; Nc=37;
[signal,if]=anafsk(N,Nc,Nbf);
if any(abs(abs(signal)-1)>sqrt(eps))~=0,
  error('anafsk test 5 failed');
end

Nbf=7; Nc=12;
[signal,if]=anafsk(N,Nc,Nbf);
if any(abs(abs(signal)-1)>sqrt(eps))~=0,
  error('anafsk test 6 failed');
end

N=211;

% Output frequency law
% At each frequency shift, on point of the iflaw is moved 
Nbf=10; Nc=32;
[signal,if]=anafsk(N,Nc,Nbf);
iflaw=instfreq(signal);
if sum(abs(iflaw-if(2:N-1))>sqrt(eps))>ceil((N-Nc)/Nc),
  error('anafsk test 7 failed');
end

Nbf=5; Nc=41;
[signal,if]=anafsk(N,Nc,Nbf);
iflaw=instfreq(signal);
if sum(abs(iflaw-if(2:N-1))>sqrt(eps))>ceil((N-Nc)/Nc),
  error('anafsk test 8 failed');
end

Nbf=15; Nc=57;
[signal,if]=anafsk(N,Nc,Nbf);
iflaw=instfreq(signal);
if sum(abs(iflaw-if(2:N-1))>sqrt(eps))>ceil((N-Nc)/Nc),
  error('anafsk test 9 failed');
end

% Output amplitude
Nbf=22; Nc=26;
[signal,if]=anafsk(N,Nc,Nbf);
if any(abs(abs(signal)-1)>sqrt(eps))~=0,
  error('anafsk test 10 failed');
end

Nbf=3; Nc=37;
[signal,if]=anafsk(N,Nc,Nbf);
if any(abs(abs(signal)-1)>sqrt(eps))~=0,
  error('anafsk test 11 failed');
end

Nbf=7; Nc=12;
[signal,if]=anafsk(N,Nc,Nbf);
if any(abs(abs(signal)-1)>sqrt(eps))~=0,
  error('anafsk test 12 failed');
end
