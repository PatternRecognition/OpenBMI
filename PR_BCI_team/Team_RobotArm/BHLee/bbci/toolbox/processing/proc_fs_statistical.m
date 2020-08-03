function [fv, idx, feat_score]= proc_fs_statistical(fv, retain, varargin)
%[fv, idx, feat_score]= proc_fs_statistical(fv, retain, <opt>)
%
% IN  fv     - struct of feature vectors
%     retain - threshold for determining how many features to retain,
%              depends on opt.policy
%     opt    propertylist or struct of options:
%      .policy - one of 'number_of_features', 'perc_of_features',
%                'perc_of_score': determines the strategy how to choose
%                the number of features to be selected
%      .method - method that is used to calculate the feature score,
%                one of 't_scale', 'r_values', 'r_square' (default),
%                'fisherScore', 'rocAreaValues'.
%
% OUT  fv    - struct of reduced feature vectors
%      idx   - indices of selected features
%      feat_score - score corresponding to idx

% kraulem 07/07/04


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'policy', 'perc_of_score', ...
                  'method', 'r_square');

if size(fv.y,1)~=2
  error('This feature selection requires exactly two classes.');
end
if isfield(fv, 'clab'),
  fv= rmfield(fv, 'clab');
end
fv= proc_flaten(fv);

fv_score= feval(['proc_' opt.method], fv);
score= abs(fv_score.x);

[so,si]= sort(-score);
so= -so;
nFeats= size(fv.x, 1);
switch(opt.policy),
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
