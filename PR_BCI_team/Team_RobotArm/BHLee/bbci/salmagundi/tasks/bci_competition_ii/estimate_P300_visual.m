file= 'bci_competition_ii/albany_P300_train';
file_test= 'bci_competition_ii/albany_P300_test';
lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 
cd([BCI_DIR 'tasks/bci_competition_ii']);
warning off bbci:validation


model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
xTrials= [100 10];
%xTrials= [10 10];
policy= 'mean';
nRep= 15;


clear cnt Epo
[cnt, mrk, mnt]= loadProcessedEEG(file);
Epo= makeEpochs(cnt, mrk, [-50 550]);

clear epo;
epo= proc_albanyAverageP300Trials(Epo, 1, nRep);
fv_col= proc_selectEpochs(epo, find(epo.code<=6));
fv_col= proc_selectChannels(fv_col, 'P7-3','P4-8','PO#','O#','Iz');
fv_col= proc_baseline(fv_col, [150 220]);
fv_col= proc_selectIval(fv_col, [220 550]);
fv_col= proc_jumpingMeans(fv_col, 10);

fv_row= proc_selectEpochs(epo, find(epo.code>6));
fv_row= proc_selectChannels(fv_row, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv_row= proc_baseline(fv_row, [0 150]);
fv_row= proc_selectIval(fv_row, [240 460]);
fv_row= proc_jumpingMeans(fv_row, 10);

Fv= makeLetterTrials_colrow(fv_col, fv_row);
P= {Fv.nRep, Fv.nF_col, Fv.nC_col, Fv.nF_row, Fv.nC_row};

%classy= {'P300_colrow', P{:}, policy, 'LDA'};
classy_col= selectModel(fv_col, model, xTrials);
classy_row= selectModel(fv_row, model, xTrials);
classy= {'P300_colrow', P{:}, policy, classy_col, classy_row};

C= trainClassifier(Fv, classy);



clear epo Epo
[cnt, mrk, mnt]= loadProcessedEEG(file_test);
Epo= makeEpochs(cnt, mrk, [-50 550]);
clear cnt

epo= proc_albanyAverageP300Trials(Epo, 1, nRep);
fv_col= proc_selectEpochs(epo, find(epo.code<=6));
fv_col= proc_selectChannels(fv_col, 'P7-3','P4-8','PO#','O#','Iz');
fv_col= proc_baseline(fv_col, [150 220]);
fv_col= proc_selectIval(fv_col, [220 550]);
fv_col= proc_jumpingMeans(fv_col, 10);

fv_row= proc_selectEpochs(epo, find(epo.code>6));
fv_row= proc_selectChannels(fv_row, 'T8,10','CP7,8','P#','PO#','O#','Iz');
fv_row= proc_baseline(fv_row, [0 150]);
fv_row= proc_selectIval(fv_row, [240 460]);
fv_row= proc_jumpingMeans(fv_row, 10);

Fv= makeLetterTrials_colrow(fv_col, fv_row);

out= applyClassifier(Fv, classy, C);
Out= lett_matrix(out)



results_file= [TEX_DIR 'bci_competition_ii/blankertz_P300_results.txt'];
fid= fopen(results_file, 'wt');
ii= 1;
OUT= [Out; repmat(' ', 1, length(Out))];
for wi= 1:length(mrk.wordLength),
  fprintf(fid, '%s |  ', OUT(:, ii:(ii+mrk.wordLength(wi)-1)));
  ii= ii+mrk.wordLength(wi);
end
fclose(fid);


results_file= [TEX_DIR 'bci_competition_ii/results.dat'];
fid= fopen(results_file, 'wt');
ii= 1;
for wi= 1:length(mrk.wordLength),
  fprintf(fid, '%s\r\n', lett_matrix(out(ii:(ii+mrk.wordLength(wi)-1))));
  ii= ii+mrk.wordLength(wi);
end
fclose(fid);

type(results_file);
