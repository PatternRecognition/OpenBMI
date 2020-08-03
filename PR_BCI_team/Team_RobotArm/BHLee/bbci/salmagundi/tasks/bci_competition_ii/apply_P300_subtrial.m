function out= apply_P300_subtrial(C, x)
%out= apply_P300_subtrial(C, x)
%
% see train_P300_subtrial

% bb, ida.first.fhg.de 04/03


nTrialsPerBlock= 12;

nRep= C.nRep;
nChans= C.nChans;
[TTT, nLett]= size(x);
T= TTT/(nChans*nRep*nTrialsPerBlock);

nSubTrials= nTrialsPerBlock*nRep*nLett;
epo.x= reshape(x, [T, nRep, nTrialsPerBlock, nChans, nLett]);
epo.x= permute(epo.x, [1 4 3 2 5]);
epo.x= reshape(epo.x, [T, nChans, nSubTrials]);

epo.code= reshape(repmat([1:12]', [1 nRep*nLett]), [1 nSubTrials]);

iRow= find(epo.code>6);
iCol= find(epo.code<=6);
out_row= applyClassifier(epo, C.classy, C.row.C, iRow);
pick_row= selectWinner(out_row, 6, nRep, C.row.policy);

out_col= applyClassifier(epo, C.classy, C.col.C, iCol);
pick_col= selectWinner(out_col, 6, nRep, C.col.policy);

out= (pick_col-1)*6 + pick_row;
