function varargout = getfromdouble(res);

n = res(1);

res = res(2:end);
for i = 1:min(nargout,n)
  [varargout{i},res] = getfrodouble(res);
end

for i = n+1:nargout
  varargout{i} = [];
end


function [A,res] = getfrodouble(res);

switch res(1)
 case 0
  [A,res] = getfromlogical(res(2:end));
 case 1
  [A,res] = getfromnumerical(res(2:end));
 case 2
  [A,res] = getfromchar(res(2:end));
 case 3
  [A,res] = getfromcell(res(2:end));
 case 4
  [A,res] = getfromstruct(res(2:end));
otherwise 
  error('format of res not assisted\n');
end



function [A,res] = getfromnumerical(res);

n = res(1);
dims = res(2:n+1);
A = res(n+2:n+1+prod(dims));
res = res(n+2+prod(dims):end);
A = reshape(A,dims);


function [A,res] = getfromlogical(res);

n = res(1);
dims = res(2:n+1);
A = res(n+2:n+1+prod(dims));
res = res(n+2+prod(dims):end);
A = logical(reshape(A,dims));

function [A,res] = getfromchar(res);

n = res(1);
dims = res(2:n+1);
A = res(n+2:n+1+prod(dims));
res = res(n+2+prod(dims):end);
A = char(reshape(A,dims));


function [A,res] = getfromcell(res);

n = res(1);
dims = res(2:n+1);
res = res(n+2:end);

A = cell(dims);
for i = 1:prod(dims);
  [A{i},res] = getfrodouble(res);
end

function [A,res] = getfromstruct(res);

n = res(1);
dims = res(2:n+1);
fi = res(n+2);
res = res(n+3:end);

for i = 1:fi
  [fie,res] = getfromchar(res(2:end));
  if i==1
    A = struct(fie,cell(dims));
  end
  
  for j = 1:prod(dims);
    [el,res] = getfrodouble(res);
    A = setfield(A,{j},fie,el);
  end
end

A = reshape(A,dims);


  
 