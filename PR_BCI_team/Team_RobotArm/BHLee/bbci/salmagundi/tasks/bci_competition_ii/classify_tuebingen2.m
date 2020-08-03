file= 'bci_competition_ii/tuebingen2_train';


[epo, mrk, mnt]= loadProcessedEEG(file);

xTrials= [10 10];
msTrials= [3 10 round(9/10*size(epo.y,2))];
model.classy= 'RLDA';
model.param= [0 0.001 0.01 0.1 0.5];
model.msDepth= 2;


fv= proc_baseline(epo, [2000 2000]);
fv= proc_selectChannels(fv, 'A2mCz');
fv= proc_selectIval(fv, [2000 6000]);
fv= proc_jumpingMeans(fv, fv.fs/2);

classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);




fv= proc_baseline(epo, [2000 2000]);
fv= proc_selectChannels(fv, 'EOGv');
fv= proc_selectIval(fv, [4000 6000]);
fv= proc_jumpingMeans(fv, fv.fs/2);

classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);
%% 43%
%doXvalidationPlus(fv, model, xTrials);


fv= proc_baseline(epo, [2000 2000]);
fv= proc_selectChannels(fv, 'A2mCz');
fv= proc_selectIval(fv, [2000 6000]);
fv= proc_jumpingMeans(fv, fv.fs/2);

fv2= proc_baseline(epo, [2000 2000]);
fv2= proc_selectChannels(fv2, 'EOGv');
fv2= proc_selectIval(fv2, [4000 6000]);
fv2= proc_jumpingMeans(fv2, fv2.fs/2);
fv.x= cat(1, fv.x, fv2.x);

classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);
%doXvalidationPlus(fv, model, xTrials);
