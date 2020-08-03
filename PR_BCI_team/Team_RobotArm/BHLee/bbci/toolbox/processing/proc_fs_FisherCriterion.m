function [fv, idx, feat_score]= proc_fs_FisherCriterion(fv, retain, varargin)
%[fv, idx, feat_score]= proc_fs_FisherCriterion(fv, retain, <policy>)
%[fv, idx, feat_score]= proc_fs_FisherCriterion(fv, retain, <opt>)
%
%WARNING: This function is obsolote: use proc_fs_statistical with
%   'method', 'fisherScore'!
%
% IN  fv     - struct of feature vectors
%     retain - threshold for determining how many features to retain,
%              depends on opt.policy
%     opt    propertylist or struct of options:
%      .policy - one of 'number_of_features', 'perc_of_features',
%                'perc_of_score': determines the strategy how to choose
%                the number of features to be selected
%
% OUT  fv    - struct of reduced feature vectors
%      idx   - indices of selected features
%      feat_score - score corresponding to idx

% kraulem 07/07/04


if length(varargin)==1 & ischar(varargin{1}),
  opt= struct('policy', varargin{1}),
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'policy', 'perc_of_score');

fv= proc_flaten(fv);
if isfield(fv, 'clab'),
  fv= rmfield(fv, 'clab');
end
if size(fv.y,1)~=2
  error('This feature selection requires exactly two classes.');
end

cl1 = find(fv.y(1,:)==1);
cl2 = find(fv.y(2,:)==1);

mu = [mean(fv.x(:,cl1),2) mean(fv.x(:,cl2),2)];
v  = [var(fv.x(:,cl1)')' var(fv.x(:,cl2)')'];

score = ((mu(:,1)-mu(:,2)).^2)./(v(:,1)+v(:,2)+eps);

[so,si]= sort(-abs(score));
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
