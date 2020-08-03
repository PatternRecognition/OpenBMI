function out= apply_c45(C, y)
%out= apply_c45(C, y)

nTrains= length(C.yTr);
out= zeros(1, size(y,2));
for ib= 1:C.nBoots,
  idx= ceil(nTrains*rand(1,nTrains));
  out= out + c45(C.xTr(:,idx), C.yTr(idx), y, C.opt);
end
out= out/C.nBoots;
