file= 'bci_competition_ii/albany_P300_train';
xTrials= [5 10];
lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 

[cnt, mrk, mnt]= loadProcessedEEG(file);
%% to save memory
cnt= proc_selectChannels(cnt, 'AFz', 'F3-4','FC3-4','C5-6','CP3-4','P7,8');

Epo= makeEpochs(cnt, mrk, [0 500]);
clear cnt
warning off bbci:validation


%policy=struct('method','ival_mean', 'param',3:12);  %% nAvg=1 only
%policy=struct('method','selected_mean', 'param',0.3);
%policy= struct('method','trimmed_mean', 'param',1.5);
%policy= 'median';
policy= 'mean';
%policy= 'vote';
%policy= 'min';

nAvg= 1;

clear epo;
epo= proc_albanyAverageP300Trials(Epo, nAvg ,1);

fv= proc_selectChannels(epo, 'AFz','F3-4','FC3-4','C5-6','CP3-4','P7,8');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 10);

Fv= makeLetterTrials(fv);
Fv= proc_flaten(Fv);
P= {Fv.nRep, length(Fv.clab)};

classy= {'P300_subtrial', P{:}, policy, 'LDA'};
[err, es, out]= doXvalidationPlus(Fv, classy, [1 1], 1);
Out= lett_matrix(out);
iErr= find(Out~=Fv.lett);
Out(iErr)= lower(Out(iErr))




model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model.param= [0 0.01 0.1 0.5 0.75];
[fv.divTr, fv.divTe]= div_letterOneOut(fv.base);
classy_st= selectModel(fv, model, []);

classy= {'P300_subtrial', P{:}, policy, classy_st{:}};
[err, es, out]= doXvalidationPlus(Fv, classy, [1 1], 1);
Out= lett_matrix(out);
iErr= find(Out~=Fv.lett);
Out(iErr)= lower(Out(iErr))
