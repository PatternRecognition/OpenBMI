function [divTr, divTe, nPick]= sample_divisionsSet(g, xTrials, setSize)
%[divTr, divTe]= sample_divisionsSet(label, xTrials)
%
% IN  label   - class labels, array of size [nClasses nSamples]
%               where row r indicates membership of class #r.
%               (0: no member, 1: member)
%     xTrials - [nShuffles nFold]
%               a third value in this vector restricts the number
%               of sets that are drawn for each fold.
%     setSize - OPTIONAL
%               size of set in which sort order should be preserved
%               If given, this parameter is passed on to
%               sampleDivisions. Default setSize is 1.
% 
% OUT divTr   - divTr{n}: cell array holding the training sets
%               folds for shuffle #n, more specificially
%               divTr{n}{m} holds the indices of the training set of
%               the m-th fold of shuffle #n.
%     divTe   - analogue to divTr, for the test sets
%

if nargin < 3
  setSize = 1;
end
  
if length(xTrials)>=3 && xTrials(3)<0,
%% set xTrials(3) to the size of the training set
  nValid= sum(any(g));
  xTrials(3)= round((xTrials(2)-1)/xTrials(2)*nValid);
end

nShuffles= xTrials(1);
nDivisions= xTrials(2);
if length(xTrials)>2,
  nPick= xTrials(3:end);
else
  nPick= 0;
end

nClasses= size(g,1);
nEpochs= size(g,2);

if(mod(nEpochs,setSize)~=0)
  error('not all Sets have setSize samples.');
end

if nPick==0,
  nPick= nEpochs/setSize;
elseif length(nPick)~=nClasses,
  error('nPick must be scalar.');
end

divTr= cell(nShuffles,1);
divTe= cell(nShuffles,1);

for ti= 1:nShuffles,
  [divTr{ti}, divTe{ti}]= sampleDivisionSet(g, nDivisions, nPick, setSize);
end
