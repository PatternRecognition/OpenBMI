% function out = apply_wr_multiClass_hsan(C,y, idx1,idx2,idx3)
function [ out, test] = apply_wr_multiClass_hsan(C,y, idx1,idx2,idx3)
applyFcn= getApplyFuncName(C.func);

test = zeros(size(C.M,2),size(y,2));

for i = 1:size(C.M,2)
    lengidx1 = length(idx1);
    lengidx2 = length(idx2);
    lengidx3 = length(idx3);

    if i == 1
        test(i,:) = feval(applyFcn, C.Cl(i),y(1:lengidx1));
    elseif i == 2
        test(i,:) = feval(applyFcn, C.Cl(i),y(lengidx1+1:lengidx1+lengidx2));
    elseif i == 3
        test(i,:) = feval(applyFcn, C.Cl(i),y(lengidx1+lengidx2+1:end));
    end
end

out = zeros(size(C.M,1),size(y,2));
for i = 1:size(C.M,1)
  out(i,:) = - sum(feval(C.coding, spdiag(C.M(i,:))*test));
end

