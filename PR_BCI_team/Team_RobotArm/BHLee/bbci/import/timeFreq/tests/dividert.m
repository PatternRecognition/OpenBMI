function dividert
%DIVIDERT Unit test for the function DIVIDER.

%	O. Lemoine - March 1996.

N=500; 

for k=1:N,
 [Nk,Mk]=divider(k);
 err(k)=abs(Nk*Mk-k);
end
				     
if any(err),
 error('divider test 1 failed');
end;
