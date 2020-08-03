%function klaudert
%KLAUDERT Unit test for function KLAUDER.

%	O. Lemoine - March 1996.

N=256;

% Admissibility condition
psi=klauder(N);
if abs(sum(psi))>sqrt(eps),
 error('klauder test 1 failed');
end

% Admissibility condition
psi=klauder(N,100,0.04);
if abs(sum(psi))>sqrt(eps),
 error('klauder test 2 failed');
end

% Admissibility condition
psi=klauder(N,1,0.49);
if abs(sum(psi))>sqrt(eps),
 error('klauder test 3 failed');
end


N=227;

% Admissibility condition
psi=klauder(N);
if abs(sum(psi))>sqrt(eps),
 error('klauder test 4 failed');
end

% Admissibility condition
psi=klauder(N,100,0.04);
if abs(sum(psi))>sqrt(eps),
 error('klauder test 5 failed');
end

% Admissibility condition
psi=klauder(N,1,0.49);
if abs(sum(psi))>sqrt(eps),
 error('klauder test 6 failed');
end
