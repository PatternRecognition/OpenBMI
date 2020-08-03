%function oddt
%ODDT 	Unit test for the function ODD.

%	O. Lemoine - August 1996.

N=1000;

X=randn(N,1)*10;

Y=odd(X);

err1=0;
err2=0;

for k=1:N,
 if (rem(Y(k),2)~=1 & rem(Y(k),2)~=-1), 
  err1=1;
 elseif abs(X(k)-Y(k))>1, 
  err2=1;
 end
end

if err1,
 error('odd test 1 failed');
elseif err2,
 error('odd test 2 failed');
end


N=987;

X=randn(N,1)*10;

Y=odd(X);

err1=0;
err2=0;

for k=1:N,
 if (rem(Y(k),2)~=1 & rem(Y(k),2)~=-1), 
  err1=1;
 elseif abs(X(k)-Y(k))>1, 
  err2=1;
 end
end

if err1,
 error('odd test 3 failed');
elseif err2,
 error('odd test 4 failed');
end