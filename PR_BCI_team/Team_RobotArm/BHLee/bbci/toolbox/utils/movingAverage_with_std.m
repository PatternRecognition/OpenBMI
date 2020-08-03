function [xa,xs]= movingAverage(x, n, method)
%[xa,xs]= movingAverage(x, n, <method='centered'>)
%
% aweful slow implementation (TODO!)

if ~exist('method', 'var'), method='centered'; end

[T,N]= size(x);
xa= zeros(T,N);

switch(method),
 case 'centered',
  w0= -ceil((n-1)/2);
  w1= n-1+w0;
  for k= 1:T,
    k0= max(1, k+w0);
    k1= min(T, k+w1);
    xa(k,:)= mean(x(k0:k1,:), 1);
    if nargout>1,
      xs(k,:)= std(x(k0:k1,:));
    end
  end
  
  case 'causal',
   for k= 1:T,
    k0= max(1, k-n+1);
    xa(k,:)= mean(x(k0:k,:), 1); 
    if nargout>1,
      xs(k,:)= std(x(k0:k1,:));
    end
   end
  
 otherwise,
  error(sprintf('method <%s> unknown', method));
end
