%function momttfrt
%MOMTTFRT Unit test for the function MOMTTFR.

%	O. Lemoine - March 1996.

N=128;

% For a perfect line
[sig,ifl]=fmlin(N);
tfr=tfrideal(ifl);
[fm,B2]=momttfr(tfr,'type2'); 
fmth=linspace(0,0.5-1/(2*N),N);
if any(abs(fm-fmth')>sqrt(eps)),
  error('momttfr test 1 failed');
elseif any(abs(B2)>sqrt(eps)),
  error('momttfr test 2 failed');
end

% For a sinusoid
[sig,ifl]=fmsin(N);
tfr=tfrideal(ifl);
[fm,B2]=momttfr(tfr,'type2'); 
if any(abs(fm-ifl)>sqrt(1/N)),
  error('momttfr test 3 failed');
elseif any(abs(B2)>sqrt(eps)),
  error('momttfr test 4 failed');
end

% For a signal composed of 3 iflaws
[sig1,ifl1]=fmlin(N); 
[sig2,ifl2]=fmsin(N);
[sig3,ifl3]=fmhyp(N,'f',[1 0.5],[N/2 0.1]); 
tfr=tfrideal([ifl1;ifl2;ifl3]); 
[fm,B2]=momttfr(tfr,'type2'); 
if any(abs(fm-[ifl1;ifl2;ifl3])>sqrt(1/N)),
  error('momttfr test 5 failed');
elseif any(abs(B2)>sqrt(eps)),
  error('momttfr test 6 failed');
end


N=111;

% For a perfect line
[sig,ifl]=fmlin(N);
tfr=tfrideal(ifl);
[fm,B2]=momttfr(tfr,'type2'); 
fmth=linspace(0,0.5-1/(2*N),N);
if any(abs(fm-fmth')>sqrt(eps)),
  error('momttfr test 7 failed');
elseif any(abs(B2)>sqrt(eps)),
  error('momttfr test 8 failed');
end

% For a sinusoid
[sig,ifl]=fmsin(N);
tfr=tfrideal(ifl);
[fm,B2]=momttfr(tfr,'type2'); 
if any(abs(fm-ifl)>sqrt(1/N)),
  error('momttfr test 9 failed');
elseif any(abs(B2)>sqrt(eps)),
  error('momttfr test 10 failed');
end

% For a signal composed of 3 iflaws
[sig1,ifl1]=fmlin(N); 
[sig2,ifl2]=fmsin(N);
[sig3,ifl3]=fmhyp(N,'f',[1 0.5],[N/2 0.1]); 
tfr=tfrideal([ifl1;ifl2;ifl3]); 
[fm,B2]=momttfr(tfr,'type2'); 
if any(abs(fm-[ifl1;ifl2;ifl3])>sqrt(1/N)),
  error('momttfr test 11 failed');
elseif any(abs(B2)>sqrt(eps)),
  error('momttfr test 12 failed');
end
