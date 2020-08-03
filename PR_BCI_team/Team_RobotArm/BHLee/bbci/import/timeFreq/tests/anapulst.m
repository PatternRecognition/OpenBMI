%function anapulst
%ANAPULST Unit test for the function ANAPULSE.

%	O. Lemoine - February 1996.

N=1024; ti=332;
sig=anapulse(N,ti);
if abs(real(sig(ti))-1)>sqrt(eps), 
 error('anapulse test 1 failed');
elseif sum(real(sig)>sqrt(eps))~=1,
 error('anapulse test 2 failed');
end

N=541;
sig=anapulse(N);
if abs(real(sig(round(N/2)))-1)>sqrt(eps), 
 error('anapulse test 3 failed');
elseif sum(real(sig)>sqrt(eps))~=1,
 error('anapulse test 4 failed');
end
