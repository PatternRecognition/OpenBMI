function epo= makeEpochsWithRest(cnt, mrk, ival, rest_shift)
% epo= makeEpochsWithRest(cnt, mrk, ival, rest_shift)

epo= makeEpochs(cnt, mrk, ival);
epo_rest= makeEpochs(cnt, mrk, ival + rest_shift);
epo_rest.y= ones(1, size(epo.y,2));
epo_rest.className= {'rest'};
epo= proc_appendEpochs(epo, epo_rest);
