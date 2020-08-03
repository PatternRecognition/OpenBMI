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
epo= proc_albanyAverageP300Trials(Epo, nAvg ,1);

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
C= train_P300_colrow(Fv.x, Fv.y, P{:}, policy, 'LDA', 'LDA');
[out, out_col, out_row, pick_col, pick_row]= apply_P300_colrow(C, Fv.x);
Fv.lett
lett_matrix(out)
train_err= 100*mean(Fv.target~=out)

out_col= reshape(out_col, [6, Fv.nRep, nLett]);
out_row= reshape(out_row, [6, Fv.nRep, nLett]);
nLett= length(Fv.target);
for il= 1:nLett,
  subplot(1, 5, 1:3);
  imagesc([out_col(:,:,il); out_row(:,:,il)]);
  subplot(1, 5, 4);
  imagesc([[1:6]==pick_col(il), [1:6]==pick_row(il)]');
  subplot(1, 5, 5);
  [ir,ic]= ind2sub([6 6], find(Fv.y(:,il)));
  imagesc([[1:6]==ic, [1:6]==ir]');
  fprintf(lett_matrix(out(il)));
  pause
end
fprintf('\n');
