function fv= proc_appendSamples(fv1, fv2)
%fv= proc_appendSamples(fv1, fv2)

nd= max(ndims(fv1.x), ndims(fv2.x));
sz1= [size(fv1.x), ones(1,nd-ndims(fv1.x))];
sz2= [size(fv2.x), ones(1,nd-ndims(fv2.x))];
if ~isequal(sz1(1:nd-1), sz2(1:nd-1)),
  error('dimension mismatch');
end

if isfield(fv1, 'className') & isfield(fv2, 'className'),
  if ~isequal(fv1.className, fv2.className),
    error('class mismatch');
  end
end
if size(fv1.y,1)~=size(fv2.y,1),
  error('numbers of classes do not match');
end


fv= copy_struct(fv1, 'not', 'x','jit','bidx');
fv.x= cat(nd, fv1.x, fv2.x);
fv.y= cat(2, fv1.y, fv2.y);

if isfield(fv1, 'jit') & isfield(fv2, 'jit'),
  fv.jit= cat(2, fv1.jit, fv2.jit);
end

if isfield(fv1, 'bidx') & isfield(fv2, 'bidx'),
  fv.bidx= cat(2, fv1.bidx, fv2.bidx);
end
