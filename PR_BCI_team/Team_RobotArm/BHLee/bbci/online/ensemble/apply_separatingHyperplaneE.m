function out= apply_separatingHyperplaneE(C, y)
%out= apply_separatingHyperplane(C, y)

ind = find(sum(abs(y),1)~=inf);

out = zeros(length(C.b),size(y,2));

y = y(:,ind);
size(y);
if size(y,1)==1
  dum= C.w'.* y;
  dum=reshape(dum,4,size(C.b',2));
  out_all(:,ind)=real(sum(dum,1) + repmat(C.b,1,size(y,2)));
  % here is the weighting
  out=C.z'*out_all;
else
  dum= repmat(C.w,1,size(y,2)) .* y;
  dum=reshape(dum,4,size(C.b,1));
  out_all(:,ind)=real(squeeze(sum(dum,1)) + repmat(C.b',1,size(y,2)));
  % here is the weighting
  out=C.z'*out_all;
end

if isfield(C,'bias')
  %size(C.bias)
  if size(C.bias,1)>1
     out_all=out_all+C.bias;
    out=C.z'*out_all;
  else
  out=out+C.bias;
  end
end
