%function midpoitt
%MIDPOITT Unit test for the function MIDPOINT.

%	O. Lemoine - February 1996.

N=256;

k=2; c=1;
t1=.5*ones(1,N); nu1=ones(1,N); t0=t1-c;
nu=(2:N+1);

% Localization on a power-law group-delay

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 1 failed');
end 

% Same thing for other values of k and c

k=1; c=1/2; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 2 failed');
end 

% Same thing for other values of k and c

k=1/2; c=10; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 3 failed');
end 

% Same thing for other values of k and c

k=0; c=-3; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 4 failed');
end 

% Same thing for other values of k and c

k=5; c=25; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 5 failed');
end 


N=235;

k=2; c=1;
t1=.5*ones(1,N); nu1=ones(1,N); t0=t1-c;
nu=(2:N+1);

% Localization on a power-law group-delay

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 6 failed');
end 

% Same thing for other values of k and c

k=1; c=1/2; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 7 failed');
end 

% Same thing for other values of k and c

k=1/2; c=10; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 8 failed');
end 

% Same thing for other values of k and c

k=0; c=-3; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 9 failed');
end 

% Same thing for other values of k and c

k=5; c=25; t0=t1-c;

tg=t0+c*nu.^(k-1);
[ti,nui]=midpoint(t1,nu1,tg,nu,k);

d=ti-(t0+c*nui.^(k-1));
if abs(sum(d))>sqrt(eps),
 error('midpoint test 10 failed');
end 

