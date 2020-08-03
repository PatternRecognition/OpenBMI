file= 'bci_competition_ii/albany_P300_train';
lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 
cd([BCI_DIR 'tasks/bci_competition_ii']);

[cnt, mrk, mnt]= loadProcessedEEG(file);
cnt= proc_selectChannels(cnt, 'T8,10','CP7,8','P#','PO#','O#','Iz');

Epo= makeEpochs(cnt, mrk, [0 550]);
clear cnt
warning off bbci:validation

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',1);
model.param= [0 0.001 0.01 0.1];
policy= 'mean';


for N= 15:-1:1,

clear epo;
epo= proc_albanyAverageP300Trials(Epo, 1, N);

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

[fv_col.divTr, fv_col.divTe]= div_letterValidation(fv_col.base, [5 10 41]);
classy_col= selectModel(fv_col, model, [], 0);
[fv_row.divTr, fv_row.divTe]= div_letterValidation(fv_row.base, [5 10 41]);
classy_row= selectModel(fv_row, model, [], 0);

classy= {'P300_colrow', P{:}, policy, classy_col, classy_row};
[err, es, out]= doXvalidationPlus(Fv, classy, [1 1]);

Out= lett_matrix(out);
iErr= find(out~=Fv.target);
Out(iErr)= lower(Out(iErr));
nErr= round(err(1)*0.42);
if nErr==1, plu=''; else plu='s'; end
fprintf(' rep=%d: \\texttt{%s}  (%d error%s)\n', N, Out, nErr, plu);

end
