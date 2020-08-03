%function izakt
%IZAKT	Unit test for the function izak.

%	O. Lemoine - February 1996.

N=256;

% Perfect reconstruction
sig=noisecg(N);
DZT=zak(sig);
sigr=izak(DZT);
errors=find(any(abs(sig-sigr)>sqrt(eps)));
if length(errors)~=0,
  error('izak test 1 failed');
end

sig=noisecg(N);
DZT=zak(sig,8,32);
sigr=izak(DZT);
errors=find(any(abs(sig-sigr)>sqrt(eps)));
if length(errors)~=0,
  error('izak test 2 failed');
end

sig=noisecg(N);
DZT=zak(sig,128,2);
sigr=izak(DZT);
errors=find(any(abs(sig-sigr)>sqrt(eps)));
if length(errors)~=0,
  error('izak test 3 failed');
end


N=315;

% Perfect reconstruction
sig=noisecg(N);
DZT=zak(sig);
sigr=izak(DZT);
errors=find(any(abs(sig-sigr)>sqrt(eps)));
if length(errors)~=0,
  error('izak test 4 failed');
end

sig=noisecg(N);
DZT=zak(sig,9,35);
sigr=izak(DZT);
errors=find(any(abs(sig-sigr)>sqrt(eps)));
if length(errors)~=0,
  error('izak test 5 failed');
end

sig=noisecg(N);
DZT=zak(sig,3,105);
sigr=izak(DZT);
errors=find(any(abs(sig-sigr)>sqrt(eps)));
if length(errors)~=0,
  error('izak test 6 failed');
end

