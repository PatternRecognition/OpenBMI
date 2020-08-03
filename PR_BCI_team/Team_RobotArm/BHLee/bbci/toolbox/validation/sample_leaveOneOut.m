function [divTr, divTe]= sample_leaveOneOut(label, xTrials)
%[divTr, divTe]= sample_leaveOneOut(label, nPick)
%
% IN  label   - class labels, array of size [nClasses nSamples]
%               where row r indicates membership of class #r.
%               (0: no member, 1: member)
%     nPick   - number of samples to be selected for doing the loo.
% 
% OUT divTr   - divTr{n}: cell array holding the training sets
%               folds for shuffle #n, more specificially
%               divTr{n}{m} holds the indices of the training set of
%               the m-th fold of shuffle #n.
%     divTe   - analogue to divTr, for the test sets

nSamples= size(label,2);
if exist('xTrials','var') && ismember(length(xTrials), [1 3]),
  nPick= xTrials(end);
  if nPick<0,  %% set nPick to the size of a loo training set
    nPick= nSamples-1;
  end
else
  nPick= nSamples;
end

divTr= cell(1, 1);
divTe= cell(1, 1);
for nn= 1:nSamples,
  divTe{1}(nn)= {nn};
  idx= randperm(nSamples);
  idx(idx==nn)= [];   %% keep it shuffled (setdiff(..,nn) wouldn't)
  divTr{1}(nn)= {idx(1:nPick-1)};
end
