function [divTr, divTe]= sample_kfold(label, xTrials, varargin)
%[divTr, divTe]= sample_kfold(label, xTrials)
%[divTr, divTe]= sample_kfold(label, xTrials, 'Parameter',Value,...)
%
% IN  label   - class labels, array of size [nClasses nSamples]
%               where row r indicates membership of class #r.
%               (0: no member, 1: member)
%     xTrials - [nShuffles nFold]
%               A third value in this vector restricts the number
%               of samples that are drawn for each fold.
%               when this third value is -1 the number of samples
%               that are chosen is set to the number samples in a
%               nFold'ed training set.
% 
% OUT divTr   - divTr{n}: cell array holding the training sets
%               folds for shuffle #n, more specificially
%               divTr{n}{m} holds the indices of the training set of
%               the m-th fold of shuffle #n.
%     divTe   - analogue to divTr, for the test sets
%
% Properties:
%  fixed_trainsamples: [1 m] vector or logical index. Use the m samples
%      with the given indices always in the training set, effectively doing
%      crossvalidation only on the remaining nSamples-m samples.  Default: []
%      (no fixed training set)
%
% NOTE sample_kfold does not care for how many samples of each class
%  are chosen each training and test sets. 
%  Use, e.g., sample_divisions to have the same ratio of
%  class members in training and test sets.
%
% Example:
%   Regression data, put all examples with label larger than 2 into the
%   trainig data (yes, this does not make sense):
%   label = [1 2 3 4];
%   [divTr,divTe] = sample_kfold(label, [1 2], 'fixed_trainsamples',label>2)
%

error(nargchk(2, inf,nargin))
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'fixed_trainsamples', []);

if ~isempty(opt.fixed_trainsamples),
  % We should put a bunch of samples always into the training set.
  if islogical(opt.fixed_trainsamples),
    if length(opt.fixed_trainsamples)~=size(label,2),
      error('Logical index <fixed_trainsamples>, size must match');
    end
    opt.fixed_trainsamples = find(opt.fixed_trainsamples);
  end
  % Pretend that we have fewer data, add the fixed training set later
  nSamples = size(label,2)-length(opt.fixed_trainsamples);
  if nSamples<xTrials(2),
    error('Not enough data left');
  end
else
  % The normal kfold sampling:
  nSamples= size(label,2);
end

if length(xTrials)<3,
  xTrials(3)= nSamples;
elseif length(xTrials)>=3 && xTrials(3)<0,
%% set xTrials(3) to the size of the training set
  xTrials(3)= round((xTrials(2)-1)/xTrials(2)*nSamples);
end

div= round(linspace(0, xTrials(3), xTrials(2)+1));
divTr= cell(xTrials(1), 1);
divTe= cell(xTrials(1), 1);
for nn= 1:xTrials(1),
  divTr{nn}= cell(xTrials(2), 1);
  divTe{nn}= cell(xTrials(2), 1);
  idx= randperm(nSamples);
  for kk= 1:xTrials(2),
    sec= div(kk)+1:div(kk+1);
    divTe{nn}{kk}= idx(sec);
    divTr{nn}{kk}= idx(setdiff(1:div(end), sec));
  end
end

if ~isempty(opt.fixed_trainsamples),
  % Now we can add the fixed set of training data:
  % Indices of those data where we do kfold on
  sampleFrom = setdiff(1:size(label,2), opt.fixed_trainsamples);
  for nn= 1:xTrials(1),
    for kk= 1:xTrials(2),
      divTe{nn}{kk} = sampleFrom(divTe{nn}{kk});
      divTr{nn}{kk} = union(opt.fixed_trainsamples, sampleFrom(divTr{nn}{kk}));
    end
  end
end
