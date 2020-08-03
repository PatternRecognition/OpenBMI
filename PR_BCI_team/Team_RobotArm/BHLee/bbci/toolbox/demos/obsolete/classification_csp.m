file= 'Gabriel_01_10_15/imagGabriel';

[cnt,mrk,mnt]= loadProcessedEEG(file);
%% exclude EMG, EOG channels
cnt= proc_selectChannels(cnt, 'not', 'E*');

band= [10 15];
[b,a]= butter(7, band/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);
epo= makeEpochs(cnt_flt, mrk, [-1000 4000]);


fv= proc_selectIval(epo, [1000 3000]);
fv.proc=['fv= proc_csp(epo, 1); ' ...
         'fv= proc_variance(fv); '];
doXvalidationPlus(fv, 'LDA', [5 10]);
%doXvalidationPlus(fv, {'RLDA', 0.1}, [5 10]);

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidation(fv, classy, [5 10]);
%% model selection within cross-validation procedure (the real thing):
%doXvalidationPlus(fv, model, xTrials);
