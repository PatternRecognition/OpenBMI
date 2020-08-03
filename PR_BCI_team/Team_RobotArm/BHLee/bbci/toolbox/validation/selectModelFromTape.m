function [classy, errMean, errStd, mn]= ...
    selectModelFromTape(fv, moTape, xTrials, verbose)
%[classy, errMean, errStd]= selectProcessFromTape(fv, moTape, xTrials, verbose)
%
% IN   fv      - data (feature) vectors
%      moTape  - name of tape with classification models
%      xTrials - number of crossvalidation trials, default [5 10]
%
% OUT  classy  - selected classification model
%      errMean - mean of crossvalidation errors [test train]
%      errStd  - std of crossvalidation errors [test train]
%      mn      - number of selected model

if ~exist('xTrials', 'var') | isempty(xTrials), xTrials=[5 10]; end
if ~exist('verbose', 'var'), verbose=0; end

if ~ischar(moTape) | ~exist(['tape_' moTape '.m'], 'file'),
  mn= 1;
  [classy, errMean, errStd]= selectModel(fv, moTape, xTrials, verbose);
  return;
end

nModels= getBlockFromTape(moTape, 0);
classy= cell(1, nModels);
errMean= zeros(nModels, 2);
errStd= zeros(nModels, 2);

for mn= 1:nModels,
  block= getBlockFromTape(moTape, mn);
  if verbose, fprintf('%s\n', block); end
  eval(block);
  [classy{mn}, errMean(mn,:), errStd(mn,:)]= ...
      selectModel(fv, model, xTrials, verbose);
end

[mi, mn]= min(errMean(:,1));
block= getBlockFromTape(moTape, mn);
if verbose, fprintf('chosen with %.1f%% error: %s\n', mi, block); end
eval(block);
classy= classy{mn};
