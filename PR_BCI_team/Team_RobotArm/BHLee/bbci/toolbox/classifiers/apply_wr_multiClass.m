function out = apply_wr_multiClass(C,y)

applyFcn= getApplyFuncName(C.func);

test = zeros(size(C.M,2),size(y,2));
for i = 1:size(C.M,2)
  test(i,:) = feval(applyFcn, C.Cl(i),y);
end

out = zeros(size(C.M,1),size(y,2));
for i = 1:size(C.M,1)
  out(i,:) = - sum(feval(C.coding, spdiag(C.M(i,:))*test));
end

