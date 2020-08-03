%function mexhatt
%MEXHATT Unit test for function MEXHAT.

%	O. Lemoine - March 1996.


% Admissibility condition
psi=mexhat;
if abs(sum(psi))>sqrt(eps),
 error('mexhat test 1 failed');
end
