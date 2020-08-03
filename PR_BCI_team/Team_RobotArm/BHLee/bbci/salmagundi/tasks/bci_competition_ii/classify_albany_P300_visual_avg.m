file= 'bci_competition_ii/albany_P300_train';
xTrials= [10 10];

[Epo, mrk, mnt]= loadProcessedEEG(file, 'avg15');

epo= proc_selectEpochs(Epo, find(Epo.code>6));
%[epo.divTr, epo.divTe]= div_letterOneOut(epo.base);
%[epo.divTr, epo.divTe]= div_letterValidation(epo.base, xTrials);
fv= proc_selectChannels(epo, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [240 460]);
%fv= proc_selectIval(fv, [220 460]);  %% 240 450
fv= proc_jumpingMeans(fv, 10);

model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy= selectModel(fv, model, [5 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%%  1.8±0.2%  (fn:  2.1±2.2%,  fp:  1.5±0.2%)  [train: 0.0±0.0%]


epo= proc_selectEpochs(Epo, find(Epo.code<=6));
fv= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv= proc_baseline(fv, [150 200]);
fv= proc_selectIval(fv, [220 550]);
fv= proc_jumpingMeans(fv, 10);
model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy= selectModel(fv, model, [5 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%%  1.2±0.3%  (fn:  2.6±0.8%,  fp:  1.0±0.4%)  [train: 0.8±0.0%]



%% this could provide additional information
fv= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [420 490]);
fv= proc_jumpingMeans(fv, 5);
classy= selectModel(fv, model, [5 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%% 9.7±0.8%


%%combine classical P300 component with visual features
epo= proc_selectEpochs(Epo, find(Epo.code>6));
fv1= proc_selectChannels(epo, 'F3-4','FC3-4','C5-6','CP3-4', 'P7,8','AFz');
fv1= proc_baseline(fv1, [0 150]);
fv1= proc_selectIval(fv1, [200 401]);
fv1= proc_jumpingMeans(fv1, 10);

fv2= proc_selectChannels(epo, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv2= proc_baseline(fv2, [0 150]);
fv2= proc_selectIval(fv2, [220 450]);
fv2= proc_jumpingMeans(fv2, 10);

fv= proc_catFeatures(fv1, fv2);
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%%  3.5±0.5%  (fn:  3.4±1.5%,  fp:  3.4±0.5%)  [train: 1.6±0.1%]


epo= proc_selectEpochs(Epo, find(Epo.code<=6));
fv1= proc_selectChannels(epo, 'F3-4','FC3-4','C5-6','CP3-4', 'P7,8','AFz');
fv1= proc_baseline(fv1, [0 150]);
fv1= proc_selectIval(fv1, [200 401]);
fv1= proc_jumpingMeans(fv1, 10);

fv2= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv2= proc_baseline(fv2, [150 200]);
fv2= proc_selectIval(fv2, [220 450]);
fv2= proc_jumpingMeans(fv2, 10);

fv= proc_catFeatures(fv1, fv2);
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%%  2.1±0.3%  (fn:  3.0±1.0%,  fp:  1.7±0.4%)  [train: 0.8±0.0%]

model= struct('classy',{{'boundErrorOfType2', 0.01, 'RLDA'}}, 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
