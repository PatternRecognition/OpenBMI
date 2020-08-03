su= 'A';
%su= 'B';

model_RLDA= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model_RLDA.param= [0 0.01 0.1 0.5];

[epo,mrk,mnt]= ...
    loadProcessedEEG([EEG_IMPORT_DIR 'bci_competition_iii/data_set_ii_' su],...
                     'ave15');
fv= proc_selectChannels(epo, 'FC1-2','C5-6','CP5-6', 'P6,3,z,4,8', ...
                        'PO7,8', 'O1');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 5);

opt_xv= struct('xTrials', [5 10], ...
               'verbosity', 2, ...
               'outer_ms', 1, ...
               'loss', {{'classwiseNormalized', sum(fv.y,2)}});

[loss, loss_std, out]= ...
    xvalidation(fv, 'LDA', opt_xv);
val_confusionMatrix(fv.y, out, 'mode','normalized');

[loss, loss_std, out]= ...
    xvalidation(fv, model_RLDA, opt_xv);
val_confusionMatrix(fv.y, out, 'mode','normalized');
