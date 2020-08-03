%function sgrpdlat
%SGRPDLAT Unit test for the function SGRPDLAY.

%	O. Lemoine - September 1996.


N=256; 

% Pulse
sig=anapulse(N);
f=linspace(0,0.5,N);
gd=sgrpdlay(sig,f);
if any(abs(gd-N/2)>sqrt(eps)),
 error('sgrpdlay test 1 failed');
end;


% Linear frequency modulation
[sig,ifl]=fmlin(N,0.05,.45);
fnorm=linspace(0.05,0.45,N);
gd=sgrpdlay(sig,fnorm)/N;
gd2=interp1(ifl,1:N,fnorm)/N;
if any(abs(gd(1:N-2)-gd2(1:N-2))>5e-2),
 error('sgrpdlay test 2 failed');
end;


% Power law group delay
[sig,gpd,f]=gdpower(N,1/2);
fnorm=linspace(0.01,0.45,108);
fnorm=fnorm(2:108);
gd=sgrpdlay(sig,fnorm);
if any(abs(gd-gpd(4:110)')/N>5e-2),
 error('sgrpdlay test 3 failed');
end;


N=237; 

% Pulse
sig=anapulse(N);
f=linspace(0,0.5,N);
gd=sgrpdlay(sig,f);
if any(abs(gd-round(N/2))>sqrt(eps)),
 error('sgrpdlay test 4 failed');
end;


% Linear frequency modulation
[sig,ifl]=fmlin(N,0.05,.45);
fnorm=linspace(0.05,0.45,N);
gd=sgrpdlay(sig,fnorm)/N;
gd2=interp1(ifl,1:N,fnorm)/N;
if any(abs(gd(2:N-6)-gd2(2:N-6))>5e-2),
 error('sgrpdlay test 5 failed');
end;


% Power law group delay
[sig,gpd,f]=gdpower(N,1/2);
fnorm=linspace(0.01,0.45,108);
fnorm=fnorm(2:108);
gd=sgrpdlay(sig,fnorm);
if any(abs(gd-gpd(4:110)')/N>5e-2),
 error('sgrpdlay test 6 failed');
end;

