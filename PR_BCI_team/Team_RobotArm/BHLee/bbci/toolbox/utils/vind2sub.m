function vecOut= vind2sub(siz, ndx)
%vecOut= vind2sub(siz, ndx)
%
% like ind2sub but returns all subindices in one vector

n = length(siz);
if n==2 & min(siz)==1,
  vecOut= ndx;
  return;
end

vecOut= zeros(1,n);
k = [1 cumprod(siz(1:end-1))];
ndx = ndx - 1;
for i = n:-1:1,
  vecOut(i) = floor(ndx/k(i))+1;
  ndx = rem(ndx,k(i));
end
