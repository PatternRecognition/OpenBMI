file= 'bci_competition_ii/albany_P300_train';
xTrials= [10 10];

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'AFz','F3-4','FC3-4','C5-6','CP3-4', ...
                              'T8,10','CP7,8','P#','PO#','O#','Iz');
Epo= makeEpochs(cnt, mrk, [-50 550]);

epo= Epo;
epo= proc_selectEpochs(epo, find(epo.code>6));
%[epo.divTr, epo.divTe]= div_letterOneOut(epo.base);
[epo.divTr, epo.divTe]= div_letterValidation(epo.base, xTrials);
fv= proc_selectChannels(epo, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [240 460]);
%fv= proc_selectIval(fv, [220 460]);  %% 240 450
fv= proc_jumpingMeans(fv, 10);

model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy= selectModel(fv, model, [5 10 round(41/42*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, [], 3);
%%  


epo= Epo;
epo= proc_selectEpochs(epo, find(epo.code<=6));
[epo.divTr, epo.divTe]= div_letterValidation(epo.base, xTrials);
fv= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv= proc_baseline(fv, [150 220]);
fv= proc_selectIval(fv, [220 550]);
fv= proc_jumpingMeans(fv, 10);
doXvalidationPlus(fv, 'LDA', [], 3);
%%

model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy= selectModel(fv, model, [5 10 round(41/42*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, [], 3);
%%  




%%combine classical P300 component with visual features - rows

epo= proc_selectEpochs(Epo, find(Epo.code>6));
[epo.divTr, epo.divTe]= div_letterValidation(epo.base, xTrials);
fv1= proc_selectChannels(epo, 'F3-4','FC3-4','C5-6','CP3-4', 'P7,8','AFz');
fv1= proc_baseline(fv1, [0 150]);
fv1= proc_selectIval(fv1, [200 401]);
fv1= proc_jumpingMeans(fv1, 10);
%doXvalidationPlus(fv1, 'LDA', [], 3);
%% 25.5%

fv2= proc_selectChannels(epo, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv2= proc_baseline(fv2, [0 150]);
fv2= proc_selectIval(fv2, [240 460]);
fv2= proc_jumpingMeans(fv2, 10);
%doXvalidationPlus(fv2, 'LDA', [], 3);
%% 16.5%

fv= proc_catFeatures(fv1, fv2);
doXvalidationPlus(fv, 'LDA');
%% 15.6%
%classy= selectModel(fv, model, [3 10 round(41/42*sum(any(fv.y)))]);
%doXvalidationPlus(fv, classy, [], 3);

fv= proc_appendFeatures(fv1, fv2);
[fv.divTr, fv.divTe]= div_letterValidation(epo.base, xTrials);
doXvalidationPlus(fv, 'probCombiner', [], 3);
%%  




%%combine classical P300 component with visual features - cols

epo= proc_selectEpochs(Epo, find(Epo.code<=6));
[epo.divTr, epo.divTe]= div_letterValidation(epo.base, xTrials);
fv1= proc_selectChannels(epo, 'AFz','F3-4','FC3-4','C5-6','CP3-4', 'P7,8');
fv1= proc_baseline(fv1, [0 150]);
fv1= proc_selectIval(fv1, [200 401]);
fv1= proc_jumpingMeans(fv1, 10);
doXvalidationPlus(fv1, 'LDA', [], 3);
%% 18.5%

fv2= proc_selectChannels(epo, 'P7-3','P4-8','PO#','O#','Iz');
fv2= proc_baseline(fv2, [150 220]);
fv2= proc_selectIval(fv2, [220 550]);
fv2= proc_jumpingMeans(fv2, 10);
doXvalidationPlus(fv2, 'LDA', [], 3);
%% 11.5%

fv= proc_catFeatures(fv1, fv2);
doXvalidationPlus(fv, 'LDA', [], 3);
%% 10%
classy= selectModel(fv, model, [3 10 round(41/42*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, [], 3);
%% 10.8%

fv= proc_appendFeatures(fv1, fv2);
[fv.divTr, fv.divTe]= div_letterValidation(epo.base, xTrials);
doXvalidationPlus(fv, 'probCombiner', [], 3);
%% 11.6%
