%function locfreqt
%LOCFREQT Unit test for the function LOCFREQ.

%	O. Lemoine - January 1996.

N=256;

% Test for a complex sinusoid
sig1=fmconst(N);
[fm1,B1]=locfreq(sig1); 
if abs(fm1-.25)>sqrt(eps),
  error('locfreq test 1 failed');
end
if abs(B1)>sqrt(eps),
  error('locfreq test 2 failed');
end

% Test for a real impulse
sig2=real(anapulse(N,N/2));
[fm2,B2]=locfreq(sig2) ;
if abs(fm2)>5e-2,
  error('locfreq test 3 failed');
end
Bth=sqrt(pi*(N^2-1)/3)/N;
if abs(B2-Bth)>sqrt(eps),
  error('locfreq test 4 failed');
end

% Test for a Gaussian window : lower bound of the Heisenber-Gabor
% inequality 
sig3=amgauss(256);
[fm3,B3]=locfreq(sig3);
[tm3,T3]=loctime(sig3); 
if abs(T3*B3-1)>sqrt(eps),
  error('locfreq test 5 failed');
end


N=231;

% Test for a complex sinusoid
sig1=fmconst(N);
[fm1,B1]=locfreq(sig1); 
if abs(fm1-.25)>5e-5,
  error('locfreq test 6 failed');
end
if abs(B1)>1e-1,
  error('locfreq test 7 failed');
end

% Test for a real impulse
sig2=real(anapulse(N,round(N/2)));
[fm2,B2]=locfreq(sig2) ;
if abs(fm2)>5e-2,
  error('locfreq test 8 failed');
end
Bth=sqrt(pi*(N^2-1)/3)/N;
if abs(B2-Bth)>sqrt(eps),
  error('locfreq test 9 failed');
end

% Test for a Gaussian window : lower bound of the Heisenber-Gabor
% inequality 
sig3=amgauss(256);
[fm3,B3]=locfreq(sig3);
[tm3,T3]=loctime(sig3); 
if abs(T3*B3-1)>sqrt(eps),
  error('locfreq test 10 failed');
end

