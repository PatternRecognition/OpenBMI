%function anaqpskt
%ANAQPSKT Unit test for the function ANAQPSK.

%	O. Lemoine - February 1996.

N=256;

% Output frequency law
% At each frequency shift, two points of the iflaw are moved 
f0=0.05; Nc=32;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 1 failed');
end

f0=0.31; Nc=17;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 2 failed');
end

f0=0.48; Nc=8;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 3 failed');
end


% Output initial phase
f0=0.05; Nc=32;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 4 failed');
end

f0=0.31; Nc=17;
[signal, pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 5 failed');
end

f0=0.48; Nc=8;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 6 failed');
end



N=211;

% Output frequency law
% At each frequency shift, two points of the iflaw are moved 
f0=0.05; Nc=32;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 7 failed');
end

f0=0.31; Nc=17;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 8 failed');
end

f0=0.48; Nc=8;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 9 failed');
end


% Output initial phase
f0=0.05; Nc=32;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 10 failed');
end

f0=0.31; Nc=17;
[signal, pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 11 failed');
end

f0=0.48; Nc=8;
[signal,pm0]=anaqpsk(N,Nc,f0);
iflaw=instfreq(signal);
if sum(abs(iflaw-f0*ones(N-2,1))>sqrt(eps))>ceil(2*(N-Nc)/Nc),
  error('anaqpsk test 12 failed');
end


