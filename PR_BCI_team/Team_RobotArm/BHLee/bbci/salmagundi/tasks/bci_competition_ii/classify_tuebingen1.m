file= 'bci_competition_ii/tuebingen1_train';


[epo, mrk, mnt]= loadProcessedEEG(file);

xTrials= [10 10];
msTrials= [3 10 round(9/10*size(epo.y,2))];
model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.001 0.01 0.1 0.5];

fv= proc_selectChannels(epo, 'A1mCz','A2mCz','C3a','C3p','C4a','C4p');
%fv= proc_correctForTimeConstant(fv, 10);
fv= proc_jumpingMeans(fv, fv.fs/2);

doXvalidation(fv, 'LDA', xTrials);

classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);
%doXvalidationPlus(fv, model, xTrials);



fv1= proc_selectChannels(epo, 'A1mCz','A2mCz');
fv2= proc_selectChannels(epo,'C3a','C3p','C4a','C4p');
fv2= proc_baseline(fv2, [2000 2200]);
fv= proc_appendChannels(fv1, fv2);
fv= proc_jumpingMeans(fv, fv.fs/2);

doXvalidation(fv, 'LDA', xTrials);

classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);
%doXvalidationPlus(fv, model, xTrials);


fv= proc_selectChannels(epo, 'not','TTD');
fv= proc_movingAverage(fv, 250);
fv= proc_jumpingMeans(fv, [2000 2500; 3000 4000; 4500 5500]);

doXvalidation(fv, 'LDA', xTrials);

classy= selectModel(fv, model, msTrials);
doXvalidation(fv, classy, xTrials);
%doXvalidationPlus(fv, model, xTrials);


band= [1 35];
fv= proc_selectChannels(epo, 'not','TTD');
fv= proc_fourierBandMagnitude(fv, band, fv.fs);
doXvalidation(fv, 'LDA', xTrials);

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidation(fv, classy, xTrials);
%doXvalidationPlus(fv, model, xTrials); %% the real thing
