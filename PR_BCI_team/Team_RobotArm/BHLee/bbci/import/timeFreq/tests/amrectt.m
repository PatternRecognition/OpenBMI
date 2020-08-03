function amrectt
%AMRECTT Unit test for the function AMRECT.

%	O. Lemoine - February 1996.


N=256; t0=149; T=50; 
sig=amrect(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amrect test 1 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=1,	 				% width
 error('amrect test 2 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amrect test 3 failed ');
end


N=120; t0=50; T=37; 
sig=amrect(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amrect test 4 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=1,					% width
 error('amrect test 5 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amrect test 6 failed ');
end


N=534; t0=354; T=101; 
sig=amrect(N,t0,T);
if abs(sig(t0)-1)>sqrt(eps),				% sig(t0)=1
 error('amrect test 7 failed ');
end
[tm,T1]=loctime(sig);
if abs(T-T1)>=1,					% width
 error('amrect test 8 failed ');	
end
dist=1:min([N-t0,t0-1]);
if any(abs(sig(t0-dist)-sig(t0+dist))>sqrt(eps))~=0, 	% symmetry
 error('amrect test 9 failed ');
end
