function amexpo2t
%AMEXPO2T Unit test for the function AMEXPO2S.

%	O. Lemoine - February 1996.


N=256; t0=149; T=50; 
sig=amexpo2s(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amexpo2s test 1 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/N),				% width
 error('amexpo2s test 2 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amexpo2s test 3 failed ');
end

N=120; t0=55; T=27; 
sig=amexpo2s(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amexpo2s test 4 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/N),				% width
 error('amexpo2s test 5 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amexpo2s test 6 failed ');
end

N=534; t0=333; T=70; 
sig=amexpo2s(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amexpo2s test 7 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/N),				% width
 error('amexpo2s test 8 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amexpo2s test 9 failed ');
end