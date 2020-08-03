%function loctimet
%LOCTIMET Unit test for the function LOCTIME.

%	O. Lemoine - January 1996.

N=256;

% Test for a real impulse
sig1=real(anapulse(N,N/2));
[tm1,T1]=loctime(sig1); 
if abs(tm1-N/2)>sqrt(eps),
  error('loctime test 1 failed');
end
if abs(T1)>sqrt(eps),
  error('loctime test 2 failed');
end

% Test for a complex sinusoid
sig2=fmconst(N);
[tm2,T2]=loctime(sig2); 
if abs(tm2-(N+1)/2)>sqrt(eps),
  error('loctime test 3 failed');
end
Tth=sqrt(3*pi*(N^2-1))/3;
if abs(T2-Tth)>sqrt(eps),
  error('loctime test 4 failed');
end

% Test for a Gaussian window : lower bound of the Heisenber-Gabor
% inequality 
sig3=amgauss(256);
[fm3,B3]=locfreq(sig3);
[tm3,T3]=loctime(sig3); 
if abs(T3*B3-1)>sqrt(eps),
  error('locfreq test 5 failed');
end


N=237;

% Test for a real impulse
sig1=real(anapulse(N,round(N/2)));
[tm1,T1]=loctime(sig1); 
if abs(tm1-round(N/2))>sqrt(eps),
  error('loctime test 6 failed');
end
if abs(T1)>sqrt(eps),
  error('loctime test 7 failed');
end

% Test for a complex sinusoid
sig2=fmconst(N);
[tm2,T2]=loctime(sig2); 
if abs(tm2-(N+1)/2)>sqrt(eps),
  error('loctime test 8 failed');
end
Tth=sqrt(3*pi*(N^2-1))/3;
if abs(T2-Tth)>sqrt(eps),
  error('loctime test 9 failed');
end

% Test for a Gaussian window : lower bound of the Heisenber-Gabor
% inequality 
sig3=amgauss(256);
[fm3,B3]=locfreq(sig3);
[tm3,T3]=loctime(sig3); 
if abs(T3*B3-1)>sqrt(eps),
  error('locfreq test 10 failed');
end



