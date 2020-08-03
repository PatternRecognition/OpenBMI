function [me,st]= meanCircRadians(x, dim)
%[me,st]= meanRadians(x, dim)

if nargin==1,
  dim= min(find(size(x)~=1));
  if isempty(dim), dim= 1; end
end

se= sum(exp(i*x), dim);
me= angle(se);

if nargout>1,
  st= abs(se)/size(x,dim);
end
