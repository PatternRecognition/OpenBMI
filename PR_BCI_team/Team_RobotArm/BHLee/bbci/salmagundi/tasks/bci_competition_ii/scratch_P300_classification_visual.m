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
policy= 'mean';
%policy= 'vote';
%policy= 'min';

nAvg= 1;

clear epo;
epo= proc_albanyAverageP300Trials(Epo, nAvg, 15);

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

classy= {'P300_colrow', P{:}, policy, 'LDA'};
[err, es, out]= doXvalidationPlus(Fv, classy, [1 1], 1);
Out= lett_matrix(out);
iErr= find(Out~=Fv.lett);
Out(iErr)= lower(Out(iErr))



%% regularization may be needed for nAvg>2
model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];
classy_col= selectModel(fv_col, model, [3 10 round(41/42*sum(any(fv_col.y)))]);
classy_row= selectModel(fv_row, model, [3 10 round(41/42*sum(any(fv_row.y)))]);

classy= {'P300_colrow', P{:}, policy, classy_col, classy_row};
[err, es, out]= doXvalidationPlus(Fv, classy, [1 1], 1);
Out= lett_matrix(out);
iErr= find(Out~=Fv.lett);
Out(iErr)= lower(Out(iErr))

results_file= [TEX_DIR 'bci_competition_ii/leave_one_letter_out.txt'];
fid= fopen(results_file, 'wt');
ii= 1;
OUT= [Out; repmat(' ', 1, length(Out))];
for wi= 1:length(mrk.wordLength),
  fprintf(fid, '%s |  ', OUT(:, ii:(ii+mrk.wordLength(wi)-1)));
  ii= ii+mrk.wordLength(wi);
end
fclose(fid);

type(results_file);
