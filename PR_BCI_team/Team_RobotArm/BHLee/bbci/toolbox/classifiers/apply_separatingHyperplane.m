function out= apply_separatingHyperplane(C, y)
%out= apply_separatingHyperplane(C, y)

ind = find(sum(abs(y),1)~=inf);

out = zeros(length(C.b),size(y,2));

y = y(:,ind);


out(:,ind)= real( C.w'*y + repmat(C.b,1,size(y,2)) );
if isfield(C,'sq') & ~isempty(C.sq)
  if size(out,1)==1
    out(:,ind) = out(:,ind)  +sum(y.*(C.sq*y));
  else
    for i=1:size(out,1)
      out(i,ind)=out(i,ind)+sum(y.*(C.sq(:,:,i)*y));
    end
  end
end

