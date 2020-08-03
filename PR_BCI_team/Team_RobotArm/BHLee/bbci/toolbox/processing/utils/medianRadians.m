function [me,st]= medianRadians(x, dim)
%[me,st]= medianRadians(x, dim)

sz= size(x);
nd= ndims(x);
if nargin==1,
  dim= min(find(sz~=1));
  if isempty(dim), dim= 1; end
end

perm= [dim, setdiff(1:nd,dim)];
sz1= sz(dim);
sz2= prod(sz(perm(2:end)));
x= permute(x, perm);
x= reshape(x, [sz1 sz2]);

tpi= 2*pi;
shift_list= -pi*[0 1/2 1 3/2];
vv= zeros(length(shift_list), sz2);
for ii= 1:length(shift_list),
  vv(ii,:)= var(mod(x+pi+shift_list(ii), tpi));
end
[mm,mi]= min(vv);
shift= zeros([1 sz2]);
for ii= 2:length(shift_list),
  shift(find(mi==ii))= shift_list(ii);
end

Shift= repmat(shift, [sz1 1]);
me0= median(-pi+mod(x+pi+Shift, tpi), 1);
shift= shift - me0;
Shift= Shift - repmat(me0, [sz1 1]);

me= median(-pi+mod(x+pi+Shift, tpi), 1);
me= -pi + mod(me+pi-shift, tpi);

if nargout>1,
  st= std(mod(x+pi+Shift, tpi));
  st= reshape(st, [1 sz(perm(2:end))]);
  st= ipermute(st, perm);
end

me= reshape(me, [1 sz(perm(2:end))]);
me= ipermute(me, perm);
