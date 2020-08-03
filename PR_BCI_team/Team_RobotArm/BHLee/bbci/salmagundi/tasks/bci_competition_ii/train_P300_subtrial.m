function C= train_P300_subtrial(xTr, yTr, nRep, nChans, policy, ...
                                classy, varargin)
%C= train_P300_subtrial(xTr, yTr, nRep, nChans, policy, classy, <params>)
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


if ~exist('policy','var'), policy='mean'; end
if isstruct(policy) & isfield(policy, 'row'),
  row_policy= policy.row;
  col_policy= policy.col;
else
  row_policy= policy;
  col_policy= policy;
end
if ~isempty(varargin),
  classy= {classy, varargin{:}};
end

nTrialsPerBlock= 12;

[TTT, nLett]= size(xTr);
T= TTT/(nChans*nRep*nTrialsPerBlock);

nSubTrials= nTrialsPerBlock*nRep*nLett;
epo.x= reshape(xTr, [T, nRep, nTrialsPerBlock, nChans, nLett]);
epo.x= permute(epo.x, [1 4 3 2 5]);
epo.x= reshape(epo.x, [T, nChans, nSubTrials]);
epo.y= zeros(1, nSubTrials);
yLett= [1:36]*yTr;
iv= 1:nTrialsPerBlock*nRep;
for il= 1:nLett
  yy= zeros(nTrialsPerBlock,1);
  yy(floor((yLett(il)-1)/6)+1)= 1;
  yy(mod(yLett(il)-1,6)+7)= 1;
  epo.y(iv)= repmat(yy, [1, nRep]);
  iv= iv+nTrialsPerBlock*nRep;
end
epo.y= [epo.y==1; epo.y==0];
epo.code= reshape(repmat([1:12]', [1 nRep*nLett]), [1 nSubTrials]);

iRow= find(epo.code>6);
iCol= find(epo.code<=6);
C.row.C= trainClassifier(epo, classy, iRow);
C.col.C= trainClassifier(epo, classy, iCol);
C.row.policy= row_policy;
C.col.policy= col_policy;
C.classy= classy;
C.nRep= nRep;
C.nChans= nChans;
