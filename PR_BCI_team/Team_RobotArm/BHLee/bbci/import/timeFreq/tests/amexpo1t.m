function amexpo1t
%AMEXPO1T Unit test for the function AMEXPO1S.

%	O. Lemoine - February 1996.


N=256; t0=32; T=50; 
sig=amexpo1s(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),			% sig(t0)=1
 error('amexpo1s test 1 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/(N-t0)),				% width
 error('amexpo1s test 2 failed ');
elseif any(abs(sig(1:t0-1))>sqrt(eps))~=0,	% null before t0
 error('amexpo1s test 3 failed ');
end

N=133; t0=28; T=15; 
sig=amexpo1s(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),			% sig(t0)=1
 error('amexpo1s test 4 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/(N-t0)),			% width
 error('amexpo1s test 5 failed ');
elseif any(abs(sig(1:t0-1))>sqrt(eps))~=0,	% null before t0
 error('amexpo1s test 6 failed ');
end

N=529; t0=409; T=31; 
sig=amexpo1s(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),			% sig(t0)=1
 error('amexpo1s test 7 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/(N-t0)),			% width
 error('amexpo1s test 8 failed ');
elseif any(abs(sig(1:t0-1))>sqrt(eps))~=0,	% null before t0
 error('amexpo1s test 9 failed ');
end
