function epo = proc_normalizeTrials(epo);

n = size(epo.x);
dat = reshape(epo.x,[n(1)*n(2), n(3)]);

dat = dat./repmat(sqrt(sum(dat.*dat,1)),[size(dat,1),1]);

epo.x = reshape(dat,n);
