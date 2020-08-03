file= 'bci_competition_ii/albany_P300_train';
xTrials= [5 10];
lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 

[cnt, mrk, mnt]= loadProcessedEEG(file);

Epo= makeEpochs(cnt, mrk, [-50 550]);
clear cnt
warning off bbci:validation


%policy=struct('method','selected_mean', 'param',0.3);
%policy= struct('method','trimmed_mean', 'param',1.5);
policy= 'median';
%policy= 'mean';
%policy= 'vote';
%policy= 'min';

nAvg= 15;

clear epo;
epo= proc_albanyAverageP300Trials(Epo, nAvg ,1);

fv= proc_selectChannels(epo, 'AFz','F3-4','FC3-4','C5-6','CP3-4','P7,8');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 10);

Fv= makeLetterTrials(fv);
Fv= proc_flaten(Fv);


param= {Fv.nRep, length(Fv.clab), policy};

classy_st= {'P300_subtrial', param{:}, 'RLDA'};
model= struct('classy',{classy_st}, 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];

classy= selectModel(Fv, model, [1 1]);
C= train_P300_subtrial(Fv.x, Fv.y, classy{2:end});
out= apply_P300_subtrial(C, Fv.x);

Fv.lett
lett_matrix(out)
train_err= 100*mean(Fv.target~=out)



doXvalidationPlus(Fv, classy, [1 1], 1);
