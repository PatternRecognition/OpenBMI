function y = nanmedian(x,dim)
%NANMEDIAN - Median value, ignoring NaN values.
%
%See median

if nargin==1,
  dim = min(find(size(x)~=1));
  if isempty(dim), dim = 1; end
end
if isempty(x), y = []; return, end

siz= size(x);
n= siz(dim);

%% Permute and reshape so that DIM becomes the row dimension of a 2-D array
perm= [dim:max(length(size(x)),dim) 1:dim-1];
x= reshape(permute(x,perm),n,prod(siz)/n);

%% Do it columnwise (unefficient)
y= zeros(1, size(x,2));
for ii= 1:size(x,2),
  idx= find(~isnan(x(:,ii)));
  if isempty(idx),
    y(ii)= NaN;
  else
    y(ii)= median(x(idx,ii));
  end
end

%% Permute and reshape back
siz(dim)= 1;
y= ipermute(reshape(y,siz(perm)),perm);
