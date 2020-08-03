function dopplert
%DOPPLERT Unit test for the function DOPPLER.

%	O. Lemoine - February 1996.

N=128; Fs=100; D=12; 

% Pure tone for a fixed target
F0=25; V=0;
[fm,am,iflaw]=doppler(N,Fs,F0,D,V);
d1=abs(fft(fm));
d2=abs(am-am(1));
d3=abs(iflaw-iflaw(1));
if sum(any(d1>sqrt(eps)))~=1,
 error('doppler test 1 failed');
end;
if sum(any(d2>sqrt(eps)))~=0,
 error('doppler test 2 failed');
end;
if sum(any(d3>sqrt(eps)))~=0,
 error('doppler test 3 failed');
end;

% Null signal for a non-emitting target
F0=0; V=50;
[fm,am,iflaw]=doppler(N,Fs,F0,D,V);
d1=abs(fm-fm(1));
d2=abs(am-am(1));
d3=abs(iflaw-iflaw(1));
if sum(any(d1>sqrt(eps)))~=0,
 error('doppler test 4 failed');
end;
if sum(any(d2>sqrt(eps)))~=0,
 error('doppler test 5 failed');
end;
if sum(any(d3>sqrt(eps)))~=0,
 error('doppler test 6 failed');
end;

% Symmetry of the amplitude modulation
F0=15.6; V=32.3; T0=52;
[fm,am,iflaw]=doppler(N,Fs,F0,D,V,T0);
dist=1:min([N-T0,T0-1]);
if any(abs(am(T0-dist)-am(T0+dist))>sqrt(eps))~=0,
 error('doppler test 7 failed');
end;


N=123; Fs=61; D=7; 

% Pure tone for a fixed target
F0=26; V=0;
[fm,am,iflaw]=doppler(N,Fs,F0,D,V);
d1=abs(fft(fm));
d2=abs(am-am(1));
d3=abs(iflaw-iflaw(1));
if sum(any(d1>sqrt(eps)))~=1,
 error('doppler test 8 failed');
end;
if sum(any(d2>sqrt(eps)))~=0,
 error('doppler test 9 failed');
end;
if sum(any(d3>sqrt(eps)))~=0,
 error('doppler test 10 failed');
end;

% Null signal for a non-emitting target
F0=0; V=47;
[fm,am,iflaw]=doppler(N,Fs,F0,D,V);
d1=abs(fm-fm(1));
d2=abs(am-am(1));
d3=abs(iflaw-iflaw(1));
if sum(any(d1>sqrt(eps)))~=0,
 error('doppler test 11 failed');
end;
if sum(any(d2>sqrt(eps)))~=0,
 error('doppler test 12 failed');
end;
if sum(any(d3>sqrt(eps)))~=0,
 error('doppler test 13 failed');
end;

% Symmetry of the amplitude modulation
F0=15.6; V=32.3; T0=51;
[fm,am,iflaw]=doppler(N,Fs,F0,D,V,T0);
dist=1:min([N-T0,T0-1]);
if any(abs(am(T0-dist)-am(T0+dist))>sqrt(eps))~=0,
 error('doppler test 14 failed');
end;

