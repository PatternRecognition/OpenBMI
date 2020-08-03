function xa= movingAverageCausal(x, n, filter_type)
% xMa = movingAverageCausal(x, n, <filter_type>)
% 
% <filter_type> can be 'mean' (default) or 'median'

if nargin < 3
    filter_type = 'mean';
end

switch lower(filter_type)
    case 'mean'
        filter_func = @mean;
    case 'median'
        filter_func = @median;
    otherwise
        error('filter_type must be either mean or median!')
end

[T,N]= size(x);
xa= zeros(T,N);
for k= 1:T,
  k0= max(1, k-n+1);
  xa(k,:)= filter_func(x(k0:k,:), 1); 
end
