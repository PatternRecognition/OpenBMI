%function fmhypt
%FMHYPT	Unit test for the function FMHYP.

%	O. Lemoine - February 1996.

N=256;
t=1:N;

% For a hyperbolic instantaneous frequency, mentionning F0 and C
F0=0.05; C=.3;
[x,iflaw]=fmhyp(N,'F',[F0,C]);
if any(abs(iflaw-(F0+C./t)')>sqrt(eps))~=0,
  error('fmhyp test 1 failed');
end

% For a hyperbolic group delay, mentionning T0 and C
T0=-1; C=.5;
[x,iflaw]=fmhyp(N,'T',[T0,C]);
if any(abs(iflaw-(C./abs(t-T0)'))>sqrt(eps))~=0,
  error('fmhyp test 2 failed');
end

% For a hyperbolic instantaneous frequency, mentionning P1 and P2
P1=[1,.1]; P2=[200,.4];
[x,iflaw]=fmhyp(N,'F',P1,P2);
if abs(iflaw(P1(1))-P1(2))>sqrt(eps) | abs(iflaw(P2(1))-P2(2))>sqrt(eps),
  error('fmhyp test 3 failed');
end

% For a hyperbolic group delay, mentionning P1 and P2
P1=[10,.45]; P2=[236,.25];
[x,iflaw]=fmhyp(N,'T',P1,P2);
if abs(iflaw(P1(1))-P1(2))>sqrt(eps) | abs(iflaw(P2(1))-P2(2))>sqrt(eps),
  error('fmhyp test 4 failed');
end


N=251;
t=1:N;

% For a hyperbolic instantaneous frequency, mentionning F0 and C
F0=0.05; C=.3;
[x,iflaw]=fmhyp(N,'F',[F0,C]);
if any(abs(iflaw-(F0+C./t)')>sqrt(eps))~=0,
  error('fmhyp test 5 failed');
end

% For a hyperbolic group delay, mentionning T0 and C
T0=-1; C=.5;
[x,iflaw]=fmhyp(N,'T',[T0,C]);
if any(abs(iflaw-(C./abs(t-T0)'))>sqrt(eps))~=0,
  error('fmhyp test 6 failed');
end

% For a hyperbolic instantaneous frequency, mentionning P1 and P2
P1=[1,.1]; P2=[200,.4];
[x,iflaw]=fmhyp(N,'F',P1,P2);
if abs(iflaw(P1(1))-P1(2))>sqrt(eps) | abs(iflaw(P2(1))-P2(2))>sqrt(eps),
  error('fmhyp test 7 failed');
end

% For a hyperbolic group delay, mentionning P1 and P2
P1=[10,.45]; P2=[236,.25];
[x,iflaw]=fmhyp(N,'T',P1,P2);
if abs(iflaw(P1(1))-P1(2))>sqrt(eps) | abs(iflaw(P2(1))-P2(2))>sqrt(eps),
  error('fmhyp test 8 failed');
end
