function amtriant
%AMTRIANT Unit test for the function AMTRIANG.

%	O. Lemoine - February 1996.


N=256; t0=149; T=50; 
sig=amtriang(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amtriang test 1 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/N),				% width
 error('amtriang test 3 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amtriang test 3 failed ');
end


N=12; t0=5; T=7; 
sig=amtriang(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amtriang test 4 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/N),				% width
 error('amtriang test 5 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amtriang test 6 failed ');
end


N=535; t0=354; T=101; 
sig=amtriang(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amtriang test 7 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=sqrt(1/N),				% width
 error('amtriang test 8 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amtriang test 9 failed ');
end
