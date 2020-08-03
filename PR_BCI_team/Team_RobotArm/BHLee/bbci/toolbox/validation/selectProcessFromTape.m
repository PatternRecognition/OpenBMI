function [pn, classy, errMean, errStd, emMean, emStd]= ...
    selectProcessFromTape(epo, prTape, model, xTrials, verbose)
%[pn, classy, errMean, errStd, emMean, emStd]= ...
%   selectProcessFromTape(fv, prTape, model, xTrials, verbose)
%
% IN   fv      - data (feature) vectors
%      prTape  - name of tape with preprocessings
%      model   - (model of a) classifier, or tape with such models
%      xTrials - number of crossvalidation trials, default [5 10]
%
% OUT  pn      - number of preprocessing with lowest test error
%      classy  - selected classifier
%      errMean - mean of crossvalidation errors [test train]
%      errStd  - std of crossvalidation errors [test train]
%      emMean  - cell array containing means of cv-errors in model selection
%      emStd   - cell array containing stds of cv-errors in model selection

if ~exist('xTrials', 'var') | isempty(xTrials), xTrials=[5 10]; end
if ~exist('verbose', 'var'), verbose=0; end

nProcs= getBlockFromTape(prTape, 0);
errMean= zeros(nProcs, 2);
errStd= zeros(nProcs, 2);

for pn= 1:nProcs,
  block= getBlockFromTape(prTape,pn);
  if verbose, fprintf('%s\n', block); end
  eval(block);
  [classy{pn}, emMean{pn}, emStd{pn}, mn]= ...
      selectModelFromTape(fv, model, xTrials, verbose);
  errMean(pn,:)= emMean{pn}(mn,:);
  errStd(pn,:)= emStd{pn}(mn,:);
end

[mi, pn]= min(errMean(:,1));
block= getBlockFromTape(prTape,pn);
if verbose, fprintf('chosen with %.1f%% error: %s\n', mi, block); end
eval(block);
classy= classy{pn};
