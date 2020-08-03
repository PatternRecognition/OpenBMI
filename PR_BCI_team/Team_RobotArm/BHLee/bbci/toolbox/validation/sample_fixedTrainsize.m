function [divTr, divTe]= sample_fixedTrainsize(label, xTrials, trainSize, varargin)
%[divTr, divTe]= sample_fixedTrainsize(label, xTrials, trainSize)
%[divTr, divTe]= sample_fixedTrainsize(label, xTrials, trainSize, 'Parameter',Value,...)
%
% IN  label     - class labels, array of size [nClasses nSamples]
%                 where row r indicates membership of class #r.
%                 (0: no member, 1: member)
%     xTrials   - nDraws
%                 if nTrials is a vector, further components are ignored
%     trainSize - in each draw a training set holding trainSize elements
%                 is sampled. if trainSize is a vector (length = #classes)
%                 from each class a corresponding number of samples is
%                 chosen.
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
%      sampling only from the remaining nSamples-m samples.  Default: []
%      (no fixed training set)
%      With fixed_trainSamples given, the size of each training set is
%      m + trainSize
%

error(nargchk(3, inf, nargin));
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'fixed_trainsamples', []);

if ~isempty(opt.fixed_trainsamples),
  % We should put a bunch of samples always into the training set:
  if islogical(opt.fixed_trainsamples),
    if length(opt.fixed_trainsamples)~=size(label,2),
      error('Logical index <fixed_trainsamples>, size must match');
    end
    opt.fixed_trainsamples = find(opt.fixed_trainsamples);
  end
  sampleFrom = setdiff(1:size(label,2), opt.fixed_trainsamples);
  label = label(:,sampleFrom);
end

nClasses= size(label, 1);

if nClasses>1 
  if (length(trainSize)==1), %% this is brute-force, sorry
    % Trying to reproduce the old behaviour for length(trainSize)==1
    label= any(label);
    nClasses = 1;
  end
else
  % 1-class problem (regression): everything belongs to class 1, so that
  % we don't need to change the sampling routine below
  label = ones(size(label));
end

divTr= cell(xTrials(1), 1);
divTe= cell(xTrials(1), 1);
for nn= 1:xTrials(1),
  idxTr= [];
  idxTe= [];
  for cc= 1:nClasses,
    idxCl= find(label(cc,:));
    idxCl= idxCl(randperm(length(idxCl)));
    idxTr= [idxTr, idxCl(1:trainSize(cc))];
    idxTe= [idxTe, idxCl(trainSize(cc)+1:end)];
  end
  divTr{nn}= {idxTr};
  divTe{nn}= {idxTe};
end

if ~isempty(opt.fixed_trainsamples),
  % Now we can add the fixed set of training data:
  for nn= 1:xTrials(1),
    divTe{nn}{1} = sampleFrom(divTe{nn}{1});
    divTr{nn}{1} = union(opt.fixed_trainsamples, sampleFrom(divTr{nn}{1}));
  end
end
