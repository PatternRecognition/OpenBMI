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

fv= proc_selectChannels(epo, 'AFz','F3-4','FC3-4','C5-6','CP3-4','P7,8');
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 10);

Fv= makeLetterTrials(fv);
Fv= proc_flaten(Fv);


C= train_P300_subtrial(Fv.x, Fv.y, Fv.nRep, length(Fv.clab), policy, 'LDA');
out= apply_P300_subtrial(C, Fv.x);
Fv.lett
lett_matrix(out)
train_err= 100*mean(Fv.target~=out)



nTrialsPerBlock= 12;
nRep= C.nRep;
[TTT, nLett]= size(Fv.x);
T= TTT/(C.nChans*nRep*nTrialsPerBlock);

nSubTrials= nTrialsPerBlock*C.nRep*nLett;
FV= Fv;
FV.x= reshape(FV.x, [T, nRep, nTrialsPerBlock, C.nChans, nLett]);
FV.x= permute(FV.x, [1 4 3 2 5]);
FV.x= reshape(FV.x, [T, C.nChans, nSubTrials]);

FV.code= reshape(repmat([1:12]', [1 nRep*nLett]), [1 nSubTrials]);

iRow= find(FV.code>6);
iCol= find(FV.code<=6);
out_row= applyClassifier(FV, C.classy, C.row.C, iRow);
pick_row= selectWinner(out_row, 6, nRep, C.row.policy);
out_col= applyClassifier(FV, C.classy, C.col.C, iCol);
pick_col= selectWinner(out_col, 6, nRep, C.col.policy);
out= (pick_col-1)*6 + pick_row;

out_col= reshape(out_col, [6, nRep, nLett]);
out_row= reshape(out_row, [6, nRep, nLett]);
for il= 1:nLett,
  subplot(1, 5, 1:3);
  imagesc([out_col(:,:,il); out_row(:,:,il)]);
  subplot(1, 5, 4);
  imagesc([[1:6]==pick_col(il), [1:6]==pick_row(il)]');
  subplot(1, 5, 5);
  [ir,ic]= ind2sub([6 6], find(FV.y(:,il)));
  imagesc([[1:6]==ic, [1:6]==ir]');
  fprintf(lett_matrix(out(il)));
  pause
end
fprintf('\n');
