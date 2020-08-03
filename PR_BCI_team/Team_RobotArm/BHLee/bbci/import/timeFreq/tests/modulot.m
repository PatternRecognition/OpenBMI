%function modulot
%MODULOT Unit test for the function modulo.

%	O. Lemoine - February 1996.

X=(-100:100);
N=13;
Y=modulo(X,N);

% Test the output values between 1 and N.
errors=find(any(Y<1));
if length(errors)~=0,
  error('modulo test 1 failed');
end

errors=find(any(Y>N));
if length(errors)~=0,
  error('modulo test 2 failed');
end

% Test the identity if X between 1 and N
X=1:N;
Y=modulo(X,N);
errors=find(any(Y-X));
if length(errors)~=0,
  error('modulo test 3 failed');
end

% Test for a vector of complex non-integer values
X=noisecg(300);
N=23.6;
Y=modulo(X,N);
errors=find(any(Y<=0));
if length(errors)~=0,
  error('modulo test 4 failed');
end
