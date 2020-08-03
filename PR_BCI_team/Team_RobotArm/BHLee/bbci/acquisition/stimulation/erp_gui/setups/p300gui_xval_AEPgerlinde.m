features= proc_jumpingMeans(epo, opt.selectival);
[features, opt.meanOpt] = proc_subtractMean(proc_flaten(features));
[features, opt.normOpt] = proc_normalize(features);
if do_xval,
    opt_xv= strukt('xTrials', [5 5], 'loss','classwiseNormalized');
    [dum,dum,outTe] = xvalidation(features, opt.model, opt_xv);
    me= val_confusionMatrix(features, outTe, 'mode','normalized');
    remainmessage = sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f\n',100*me(1,1),100*me(2,2));
    xval_result = [100*me(1,1),100*me(2,2)];
end