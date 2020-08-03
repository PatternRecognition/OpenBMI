file= 'bci_competition_ii/albany_P300_train';
xTrials= [5 10];

[cnt, mrk, mnt]= loadProcessedEEG(file);

epo= makeEpochs(cnt, mrk, [-50 550]);
epo.code= mrk.code;
epo= proc_albanyAverageP300Trials(epo, 15);

fv= proc_selectChannels(epo, 'F3-4','FC3-4','C5-6','CP3-4', ...
                        'P7,8','AFz');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 10);

doXvalidationPlus(fv, 'LDA', xTrials, 3);
%%  5.3±0.5%  (fn: 12.9±1.7%,  fp:  4.0±0.6%)

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%% 3.4±0.1%  (fn:  5.3±0.0%,  fp:  3.1±0.1%)
%doXvalidationPlus(fv, model, xTrials, 3


%% this could provide additional information
fv2= proc_selectChannels(epo, 'P#','PO#','O#');   %% Iz?
fv2= proc_baseline(fv2, [0 150]);
fv2= proc_selectIval(fv2, [400 550]);
fv2= proc_jumpingMeans(fv2, 6);

doXvalidationPlus(fv2, 'LDA', xTrials, 3);
%% 16.5%

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv2, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv2, classy, xTrials, 3);
%doXvalidationPlus(fv2, model, xTrials, 3
%% 14%


fv= proc_catFeatures(fv, fv2);
model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%%  2.3±0.2%  (fn:  3.2±0.7%,  fp:  2.5±0.1%)



%% this does not seem to make sense
fv= proc_selectChannels(epo, 'F3-4','FC5-6','C5-6','CP5-6');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [600 750]);
fv= proc_jumpingMeans(fv, 6);

doXvalidationPlus(fv, 'LDA', xTrials, 3);
%% 30%

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidationPlus(fv, classy, xTrials, 3);
%doXvalidationPlus(fv, model, xTrials, 3
%% 26.5%
