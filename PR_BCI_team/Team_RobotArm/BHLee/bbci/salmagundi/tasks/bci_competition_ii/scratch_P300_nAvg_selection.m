file= 'bci_competition_ii/albany_P300_train';
xTrials= [5 10];
lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 

[cnt, mrk, mnt]= loadProcessedEEG(file);

Epo= makeEpochs(cnt, mrk, [-50 550]);
clear cnt
warning off bbci:validation


%policy=struct('method','selected_mean', 'param',0.3);
%policy= struct('method','trimmed_mean', 'param',1.5);
%policy= 'median';
%policy= 'mean';
%policy= 'vote';
%policy= 'min';

N= [1:5 7 15];
err= zeros(length(N),2);

for in= 1:length(N), fprintf('%d> ', N(in));
nAvg= N(in);

clear epo;
epo= proc_albanyAverageP300Trials(Epo, nAvg ,1);

fv= proc_selectChannels(epo, 'AFz','F3-4','FC3-4','C5-6','CP3-4','P7,8');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 10);

Fv= makeLetterTrials(fv);
Fv= proc_flaten(Fv);

classy= {'P300_subtrial', Fv.nRep, length(Fv.clab), policy, 'LDA'};
err(in,:)= doXvalidationPlus(Fv, classy, [1 1], 1);

end

