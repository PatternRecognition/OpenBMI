function C= train_P300_colrow(xTr, yTr, nRep, nF_col, nC_col, nF_row, ...
                              nC_row, policy, classy_col, classy_row)
%C= train_P300_colrow(xTr, yTr, nRep,  nF_col, nC_col, nF_row, ...
%                               nC_row, policy, classy_col, <classy_row>)
%
% IN  policy - structure with fields
%               .method: {'vote', 'min', 'mean', 'median', 
%                         'trimmed_mean', 'selected_mean'}
%               .param: parameter for policy method (if neccessary)
%              if the method does not need parameters the argument
%              may just be a string.
%              (may also be a struct with fields 'row' and 'col'
%               to choose different methods for row and col treatment)
%     params - list of parameters for the classifier
%
% OUT C      - classifier structure

% bb, ida.first.fhg.de 04/03


if ~exist('classy_row','var'), classy_row=classy_col; end
if isstruct(policy) & isfield(policy, 'row'),
  row_policy= policy.row;
  col_policy= policy.col;
else
  row_policy= policy;
  col_policy= policy;
end

nTrialsPerColrow= 6;

[TTT, nLett]= size(xTr);
T_col= nF_col*nC_col;
T_row= nF_row*nC_row;

nSubTrials= nTrialsPerColrow*nRep*nLett;
xx= reshape(xTr, [T_col+T_row, nTrialsPerColrow, nRep, nLett]);
fv_col.x= xx(1:T_col, :, :, :);
fv_col.x= reshape(fv_col.x, [nF_col, nC_col, nSubTrials]);
fv_col.y= zeros(1, nSubTrials);
fv_row.x= xx(T_col+1:end, :, :, :);
fv_row.x= reshape(fv_row.x, [nF_row, nC_row, nSubTrials]);
fv_row.y= zeros(1, nSubTrials);
yLett= [1:36]*yTr;
iv= 1:nTrialsPerColrow*nRep;
for il= 1:nLett,
  yy= zeros(nTrialsPerColrow,1);
  yy(floor((yLett(il)-1)/6)+1)= 1;
  fv_col.y(iv)= repmat(yy, [1, nRep]);
  yy= zeros(nTrialsPerColrow,1);
  yy(mod(yLett(il)-1,6)+1)= 1;
  fv_row.y(iv)= repmat(yy, [1, nRep]);
  iv= iv+nTrialsPerColrow*nRep;
end
fv_col.y= [fv_col.y==1; fv_col.y==0];
fv_row.y= [fv_row.y==1; fv_row.y==0];
fv_col.code= reshape(repmat([1:6]', [1 nRep*nLett]), [1 nSubTrials]);
fv_row.code= reshape(repmat([7:12]', [1 nRep*nLett]), [1 nSubTrials]);

C.col.C= trainClassifier(fv_col, classy_col);
C.row.C= trainClassifier(fv_row, classy_row);
C.row.policy= row_policy;
C.col.policy= col_policy;
C.classy_col= classy_col;
C.classy_row= classy_row;
C.nRep= nRep;
C.nF_col= nF_col;
C.nC_col= nC_col;
C.nF_row= nF_row;
C.nC_row= nC_row;
