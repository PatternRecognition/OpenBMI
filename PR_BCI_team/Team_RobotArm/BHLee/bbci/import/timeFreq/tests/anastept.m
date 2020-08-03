%function anastept
%ANASTEPT Unit test for the function ANASTEP.

%	O. Lemoine - February 1996.

N=1024; ti=332;
sig=anastep(N,ti);
if sum(abs(real(sig(ti:N))-1)>sqrt(eps))~=0, 
 error('anastep test 1 failed');
elseif sum(real(sig(1:ti-1))>sqrt(eps))~=0,
 error('anastep test 2 failed');
end

N=541;
sig=anastep(N);
ti=round(N/2);
if sum(abs(real(sig(ti:N))-1)>sqrt(eps))~=0, 
 error('anastep test 1 failed');
elseif sum(real(sig(1:ti-1))>sqrt(eps))~=0,
 error('anastep test 2 failed');
end
