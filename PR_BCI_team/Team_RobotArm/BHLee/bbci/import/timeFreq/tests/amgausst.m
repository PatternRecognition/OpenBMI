function amgausst
%AMGAUSST Unit test for the function AMGAUSS.

%	O. Lemoine - February 1996.


N=256; t0=149; T=50; 
sig=amgauss(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amgauss test 1 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>sqrt(eps),					% width
 error('amgauss test 2 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amgauss test 3 failed ');
end

N=121; t0=60; T=27; 
sig=amgauss(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amgauss test 4 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>sqrt(eps),					% width
 error('amgauss test 5 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amgauss test 6 failed ');
end

N=534; t0=333; T=70; 
sig=amgauss(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amgauss test 7 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>sqrt(eps),					% width
 error('amgauss test 8 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amgauss test 9 failed ');
end
