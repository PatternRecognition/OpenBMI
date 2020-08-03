%function dopnoist
%DOPNOIST Unit test for the function DOPNOISE.

%	O. Lemoine - March 1996.

N=128; Fs=100; D=12; 

% The only value we can test is iflaw since y is a random variable.

% Pure tone for a fixed target
F0=25; V=0;
[y,iflaw]=dopnoise(N,Fs,F0,D,V);
d=abs(iflaw-iflaw(1));
if sum(any(d>sqrt(eps)))~=0,
 error('dopnoise test 1 failed');
end;

% Null signal for a non-emitting target
F0=0; V=50;
[y,iflaw]=dopnoise(N,Fs,F0,D,V);
d=abs(iflaw-iflaw(1));
if sum(any(d>sqrt(eps)))~=0,
 error('dopnoise test 2 failed');
end;


N=111; Fs=51; D=9; 

% The only value we can test is iflaw since y is a random variable.

% Pure tone for a fixed target
F0=12; V=0;
[y,iflaw]=dopnoise(N,Fs,F0,D,V);
d=abs(iflaw-iflaw(1));
if sum(any(d>sqrt(eps)))~=0,
 error('dopnoise test 3 failed');
end;

% Null signal for a non-emitting target
F0=0; V=51;
[y,iflaw]=dopnoise(N,Fs,F0,D,V);
d=abs(iflaw-iflaw(1));
if sum(any(d>sqrt(eps)))~=0,
 error('dopnoise test 4 failed');
end;
