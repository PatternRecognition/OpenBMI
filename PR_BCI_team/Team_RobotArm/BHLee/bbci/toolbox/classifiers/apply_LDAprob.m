function out= apply_LDA(C, y)
%out= apply_separatingHyperplane(C, y)

ind = find(sum(abs(y),1)~=inf);

out = zeros(length(C.b),size(y,2));

y = y(:,ind);

%  out(:,ind)= real( C.w'*y + repmat(C.b,1,size(y,2)) );

y1 = y - repmat(C.mean(:, 1),1,size(y,2));
y2 = y - repmat(C.mean(:, 2),1,size(y,2));

C_chol = chol(C.invcov);

loglipri1 = log(C.prior(1)) - 0.5*sum((C_chol*y1).^2) - (size(y,1)/2)*log(2*pi) - 0.5*logdet(C.cov);
loglipri2 = log(C.prior(2)) - 0.5*sum((C_chol*y2).^2) - (size(y,1)/2)*log(2*pi) - 0.5*logdet(C.cov);
logevi = loglipri1 + log(1+exp(loglipri2-loglipri1));

out(:,ind) = exp(loglipri1 - logevi) - exp(loglipri2 - logevi);

