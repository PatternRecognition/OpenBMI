function out = apply_mixGauss(C,xTr);
%THE APPLY/FUNCTION to train_mixGauss

if length(C.k)==1 & C.k==1
  out = apply_separatingHyperplane(C,xTr);
  return
end

st = [0,cumsum(C.k)];

nTrials = size(xTr,2);

ot = zeros(sum(C.k),nTrials);
for i = 1:sum(C.k)
  ou = xTr-repmat(C.means(:,i),[1,nTrials]);
  ou = -0.5*sum(ou.*(C.covs(:,:,i)*ou),1);
  ou = exp(ou)/sqrt(C.detcovs(i));
  ot(i,:) = ou*C.priors(i);
end

out = zeros(length(C.k),nTrials);

for i = 1:length(C.k)
  out(i,:) = sum(ot(st(i)+1:st(i+1),:),1);
end

if size(out,1)==2
  out = out(2,:)-out(1,:);
end


