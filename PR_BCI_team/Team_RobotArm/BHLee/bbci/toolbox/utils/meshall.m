function cellOut= meshall(varargin)
%C= meshall(a,b,...)  % C= {A, B, ...}
%
% as ndgrid, but output arrays are arranged in a cell

N= nargin;
if N==1,
  cellOut= {full(varargin{1}(:))};
  return;
end
index= cell(N,1);
for n= 1:N,
  len= prod(size(varargin{n}));
  shape= ones(1,N);
  shape(n)= len;
  varargin{n}= reshape(full(varargin{n}(:)), shape);
  index{n}= ones(len,1);
end
cellOut= cell(1,N);
for n= 1:N,
  idx= index;
  idx{n}= ':';
  cellOut{n}= varargin{n}(idx{:});
end
