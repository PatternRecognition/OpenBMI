function [divTr, divTe, nPick]= sample_divisions(g, xTrials, skew)
%[divTr, divTe]= sample_divisions(g, xTrials)
%
% IN  g       - class labels, array of size [nClasses nSamples]
%               where row r indicates membership of class #r.
%               (0: no member, 1: member)
%     xTrials - [nShuffles nFold]
%               nFolds: the number of divisions of the data. Validation is done for each division.
%                       nFolds should not be larger than the smallest class.
%               nShuffles: the number of times the nFold validation is repeated.
%               A third value in this vector restricts the number
%               of samples that are drawn for each fold.
%               when this third value is -1 the number of samples
%               that are chosen is set to the number samples in a
%               nFold'ed training set.
%               when nShuffles=1 and nFold=1, sample_leaveOneOut is
%               called.
%     skew    - [nClasses, 1] OPTIONAL
%               If given, this parameter is passed on to
%               sampleDivisions. For each class it provides a factor
%               within [0 inf] that specifies how this class should be
%               under (<1) or over (>1) samples in the *training*
%               set. See sampleDivision() for details.
% 
% OUT divTr   - divTr{n}: cell array holding the training sets
%               folds for shuffle #n, more specificially
%               divTr{n}{m} holds the indices of the training set of
%               the m-th fold of shuffle #n.
%     divTe   - analogue to divTr, for the test sets
%
% NOTE sample_divisions draws for training and test set (approx) the
%  same ratio of class members (in contrast to sample_kfold) (modulo the
%  factor in skew, if specified).

if nargin < 3
  skew = [];
end
  
if isequal(xTrials(1:2),[1 1]),  %% leave-one-out
  [divTr, divTe]= sample_leaveOneOut(g, xTrials);
  return;
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
nEventsInClass= sum(g,2);

if min(nEventsInClass)<nDivisions,
  msg= ['number of folds greater than samples in smallest class\n' ...
        'switching to leave-one-out'];
  bbci_warning(msg, 'sample', mfilename);
  [divTr, divTe]= sample_leaveOneOut(g, [1 1 xTrials(3:end)]);
  return;  
end


if nPick==0,
  nPick= nEventsInClass;
elseif length(nPick)==1,
  totalPick= nPick;
  nPick= round(totalPick*nEventsInClass/sum(nEventsInClass));
  nPick= min([nPick; nEventsInClass]);
elseif length(nPick)~=nClasses,
  error('nPick must be scalar or match #classes');
end

divTr= cell(nShuffles,1);
divTe= cell(nShuffles,1);

for ti= 1:nShuffles,
  [divTr{ti}, divTe{ti}]= sampleDivision(g, nDivisions, nPick, skew);
end
