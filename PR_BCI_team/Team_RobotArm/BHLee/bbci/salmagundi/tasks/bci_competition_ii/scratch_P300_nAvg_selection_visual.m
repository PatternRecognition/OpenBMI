file= 'bci_competition_ii/albany_P300_train';
xTrials= [5 10];
lett_matrix= reshape(['A':'Z', '1':'9', ' '], 6, 6)'; 
cd([BCI_DIR 'tasks/bci_competition_ii']);

[cnt, mrk, mnt]= loadProcessedEEG(file);

Epo= makeEpochs(cnt, mrk, [-50 550]);
clear cnt
warning off bbci:validation

model= struct('classy','RLDA', 'msDepth',3, 'inflvar',1);
model.param= [0 0.01 0.1 0.5];

%policy= struct('method','selected_mean', 'param',0.3);
%policy= struct('method','trimmed_mean', 'param',1.5);
%policy= 'median';
policy= 'mean';
%policy= 'vote';
%policy= 'min';

best_err= zeros(15, 1);
best_out= zeros(15, 42);
best_avg= zeros(15, 1);

for N= 1:15,

facs= find(mod(N, 1:N)==0);
err= zeros(length(facs),2);
out= zeros(length(facs), 42);

for fi= 1:length(facs), 

nAvg= facs(fi);
fprintf('[%2d/%2d]> ', N, nAvg);

clear epo;
epo= proc_albanyAverageP300Trials(Epo, nAvg, N/nAvg);

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

msTrials= [5 10 round(41/42*sum(any(fv_col.y)))];
classy_col= selectModel(fv_col, model, msTrials, 0);
classy_row= selectModel(fv_row, model, msTrials, 0);

classy= {'P300_colrow', P{:}, policy, classy_col, classy_row};
[err(fi,:), es, out(fi,:)]= doXvalidationPlus(Fv, classy, [1 1], 1);

end

[best_err(N), mi]= min(err(:,1));
best_out(N,:)= out(mi,:);
best_avg(N)= facs(mi);

end


policy
for N= 15:-1:1,
  Out= lett_matrix(best_out(N,:));
  iErr= find(best_out(N,:)~=Fv.target);
  Out(iErr)= lower(Out(iErr));
  fprintf('[%2d/%2d]> %s  (%2.1f%%)\n', N, best_avg(N), Out, best_err(N));
end
