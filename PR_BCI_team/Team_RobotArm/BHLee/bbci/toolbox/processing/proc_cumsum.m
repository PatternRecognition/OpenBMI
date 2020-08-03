function epo= proc_cumsum(epo)

epo.x(:,:)= cumsum(epo.x(:,:), 1);
