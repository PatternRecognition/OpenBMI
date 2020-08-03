%function holdert
%HOLDERT Unit test for the function HOLDER.

%	O. Lemoine - August 1996.

N=256;

h=0; t0=N/2;
sig=anasing(N,t0,h);
[tfr,t,f]=tfrscalo(sig,1:N,16,0.01,0.49,2*N);
h0=holder(tfr,f,1,2*N,t0,0);
if abs(h-h0)>1e-2,
 error('holder test 1 failed');
end


h=-0.5; t0=N/2;
sig=anasing(N,t0,h);
[tfr,t,f]=tfrscalo(sig,1:N,16,0.01,0.49,2*N);
h0=holder(tfr,f,1,2*N,t0,0);
if abs(h-h0)>1e-2,
 error('holder test 2 failed');
end



N=211;

h=0; t0=(N-1)/2;
sig=anasing(N,t0,h);
[tfr,t,f]=tfrscalo(sig,1:N,16,0.01,0.49,2*N);
h0=holder(tfr,f,1,2*N,t0,0);
if abs(h-h0)>1e-2,
 error('holder test 3 failed');
end


h=-0.4; t0=(N-1)/2;
sig=anasing(N,t0,h);
[tfr,t,f]=tfrscalo(sig,1:N,16,0.01,0.49,2*N);
h0=holder(tfr,f,1,2*N,t0,0);
if abs(h-h0)>1e-2,
 error('holder test 4 failed');
end

