function epo= proc_removeEpochs(epo, idx)
%epo= proc_removeEpochs(epo, idx)

nEpochs= size(epo.x,3);
epo= proc_selectEpochs(epo, setdiff(1:nEpochs, idx));
