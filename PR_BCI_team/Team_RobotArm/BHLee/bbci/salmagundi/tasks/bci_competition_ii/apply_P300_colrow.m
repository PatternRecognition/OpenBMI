function [out, out_col, out_row, pick_col, pick_row]= apply_P300_colrow(C, x)
%out= apply_P300_colrow(C, x)
%
% see train_P300_colrow

% bb, ida.first.fhg.de 04/03


nTrialsPerColrow= 6;

nRep= C.nRep;
nF_col= C.nF_col;
nC_col= C.nC_col;
nF_row= C.nF_row;
nC_row= C.nC_row;

[TTT, nLett]= size(x);
T_col= nF_col*nC_col;
T_row= nF_row*nC_row;

nSubTrials= nTrialsPerColrow*nRep*nLett;
xx= reshape(x, [T_col+T_row, nTrialsPerColrow, nRep, nLett]);
fv_col.x= xx(1:T_col, :, :, :);
fv_col.x= reshape(fv_col.x, [nF_col, nC_col, nSubTrials]);
fv_row.x= xx(T_col+1:end, :, :, :);
fv_row.x= reshape(fv_row.x, [nF_row, nC_row, nSubTrials]);
fv_col.code= reshape(repmat([1:6]', [1 nRep*nLett]), [1 nSubTrials]);
fv_row.code= reshape(repmat([7:12]', [1 nRep*nLett]), [1 nSubTrials]);

out_col= applyClassifier(fv_col, C.classy_col, C.col.C);
pick_col= selectWinner(out_col, 6, nRep, C.col.policy);

out_row= applyClassifier(fv_row, C.classy_row, C.row.C);
pick_row= selectWinner(out_row, 6, nRep, C.row.policy);

out= (pick_col-1)*6 + pick_row;
