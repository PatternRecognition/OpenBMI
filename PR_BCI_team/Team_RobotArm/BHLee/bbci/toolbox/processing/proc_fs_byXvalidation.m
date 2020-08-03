function [fv, idx, feat_score]= proc_fs_byXvalidation(fv, retain, varargin)
%PROC_FS_BYXVALIDATION - Select Features by Cross-Validation
%
%Description
% This function offers different ways of feature selection based on
% cross-validation. The simple default method is SINGLE_FEATURE
% classification. This method can be useful if the feature are
% sufficiently independent. Otherwise INCREMENTAL or DECREMENTAL
% feature selection can be chosen. INCREMENTAL selects first the
% best classifyable single feature
% Note: For method DECREMENTAL it is strongly recommended to speficy
% a regularized classification model.
%
%Synopsis:
% [FV, IDX, FEAT_SCORE]= proc_fs_byXvalidation(FV, RETAIN, <OPT>)
%
%Arguments:
%   FV     - struct of feature vectors
%   RETAIN - threshold for determining how many features to retain,
%            depends on opt.policy
%   OPT    - propertylist or struct of options:
%    .policy - one of 'number_of_features', 'perc_of_features',
%              'perc_of_score': determines the strategy how to choose
%              the number of features to be selected
%    .model - model of the classification algorithm, see xvalidation.
%    .method - one of 'single_feature' (default), 'incremental',
%              'decremental'. The latter two methods only work with
%               OPT.policy 'number_of_features'.
%    .opt_xv - struct of optional properties that is passed to
%              xvalidation
%
%Returns:
% FV    - struct of reduced feature vectors
% IDX   - indices of selected features
% FEAT_SCORE - score corresponding to idx

% Author(s): Benjamin Blankertz


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'policy', 'number_of_features', ...
                  'model', 'LDA', ...
                  'method', 'single_feature', ...
                  'opt_xv', struct('verbosity',0));

if isfield(fv, 'clab'),
  fv= rmfield(fv, 'clab');
end
fv= proc_flaten(fv);

nFeats= size(fv.x, 1);
fv_score= ones(nFeats, 1);
ff= fv;

switch(lower(opt.method)),
 case 'single_feature',
  for ii= 1:nFeats,
    ff.x= fv.x(ii,:);
    fv_score(ii)= xvalidation(ff, opt.model, opt.opt_xv);
  end
 case 'incremental',
  idx= [];
  lasterr= 0;
  for jj= 1:retain,
    err= inf*ones(nFeats, 1);
    for ii= setdiff(1:nFeats, idx),
      ff= proc_selectFeatures(fv, [idx ii]);
      err(ii)= xvalidation(ff, opt.model, opt.opt_xv);
    end
    [mi, best_chan]= min(err);
    fv_score(best_chan)= mi-lasterr;
    lasterr= mi;
    idx= [idx, best_chan];
  end
 case 'decremental',
  idx= [];
  fv_score(:)= 0;
  for jj= 1:nFeats-retain,
    err= inf*ones(nFeats, 1);
    for ii= setdiff(1:nFeats, idx),
      ff= proc_selectFeatures(fv, setdiff(1:nFeats, [idx ii]));
      err(ii)= xvalidation(ff, opt.model, opt.opt_xv);
    end
    [mi, worst_chan]= min(err);
    fv_score(worst_chan)= 1;
    idx= [idx, worst_chan]
  end
  idx= setdiff(1:nFeats, idx);
 otherwise,
  error('method unknown');
end
score= 1-fv_score;

[so,si]= sort(-score);
so= -so;
switch(lower(opt.policy)),
 case 'number_of_features',
  idx= si(1:min(retain,nFeats));
 case 'perc_of_features',
  idx= si(1:ceil(retain/100*nFeats));
 case 'perc_of_score',
  perc= 100*cumsum(so/sum(so));
  nSel= min([find(perc>retain); nFeats]);
  idx= si(1:nSel);
 otherwise,
  error('policy unknown');
end

fv.x= fv.x(idx,:);
feat_score= so(1:nFeats);
