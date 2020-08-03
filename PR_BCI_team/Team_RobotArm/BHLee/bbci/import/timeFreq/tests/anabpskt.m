%function anabpskt
%ANABPSKT Unit test for the function ANABPSK.

%	O. Lemoine - February 1996.

N=256;

% Output frequency law
% At each frequency shift, two points of the iflaw are moved 
f0=0.05; Nc=32;
[signal,am]=anabpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anabpsk test 1 failed');
end

f0=0.31; Nc=17;
[signal,if]=anabpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anabpsk test 2 failed');
end

f0=0.48; Nc=8;
[signal,if]=anabpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anabpsk test 3 failed');
end


% Output amplitude
f0=0.01; Nc=26;
[signal,am]=anabpsk(N,Nc,f0);
if any(abs(am)-1)~=0,
  error('anabpsk test 4 failed');
end

f0=0.17; Nc=4;
[signal,am]=anabpsk(N,Nc,f0);
if any(abs(am)-1)~=0,
  error('anabpsk test 5 failed');
end

f0=0.43; Nc=43;
[signal,am]=anabpsk(N,Nc,f0);
if any(abs(am)-1)~=0,
  error('anabpsk test 6 failed');
end


N=221;

% Output frequency law
% At each frequency shift, two points of the iflaw are moved 
f0=0.05; Nc=32;
[signal,am]=anabpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anabpsk test 7 failed');
end

f0=0.31; Nc=17;
[signal,if]=anabpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anabpsk test 8 failed');
end

f0=0.48; Nc=8;
[signal,if]=anabpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anabpsk test 9 failed');
end


% Output amplitude
f0=0.01; Nc=26;
[signal,am]=anabpsk(N,Nc,f0);
if any(abs(am)-1)~=0,
  error('anabpsk test 10 failed');
end

f0=0.17; Nc=4;
[signal,am]=anabpsk(N,Nc,f0);
if any(abs(am)-1)~=0,
  error('anabpsk test 11 failed');
end

f0=0.43; Nc=43;
[signal,am]=anabpsk(N,Nc,f0);
if any(abs(am)-1)~=0,
  error('anabpsk test 12 failed');
end
