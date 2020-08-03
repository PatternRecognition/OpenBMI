function epo = proc_normalizeChannels(epo);

nn = ndims(epo.x);
epo.x = permute(epo.x,[2 1 3:ndims(epo.x)]);


% $$$ if prod(size(epo.x))<10^6
% $$$   epo.x(:,:) = epo.x(:,:)./repmat(sqrt(sum(epo.x(:,:).*epo.x(:,:),1)),[size(epo.x,1),1]);
% $$$ else
% $$$   for i = 1:size(epo.x,2)
% $$$     epo.x(:,i) = epo.x(:,i)/sqrt(epo.x(:,i)'*epo.x(:,i));
% $$$   end
% $$$ end

epo.x(:,:) = epo.x(:,:)./repmat(sqrt(sum(epo.x(:,:).*epo.x(:,:),1)),[size(epo.x,1),1]);

epo.x = permute(epo.x,[2,1,3:nn]);
