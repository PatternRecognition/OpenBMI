function [fv, idx, feat_score]= ...
    proc_fs_classifierWeights(fv, retain, model, varargin)
%[fv, idx, feat_score]= proc_fs_classifierWeights(fv, retain, model, <policy>)
%[fv, idx, feat_score]= proc_fs_classifierWeights(fv, retain, model, <opt>)
%
% IN  fv     - struct of feature vectors
%     retain - threshold for determining how many features to retain,
%              depends on opt.policy
%     model  - classification model, a sparse make most sense here;
%              default struct('classy',{{'FDlwlx','*log'}}, ...
%              'param',[-1:3], 'msDepth',2);
%     opt    propertylist or struct of options:
%      .policy - one of 'number_of_features', 'perc_of_features',
%                'perc_of_weights': detemines the strategy how to choose
%                the number of features to be selected
%      ...     - other fields are passed to the select_model,
%                which is called in case 'model' has free parameters.
%
% OUT  fv    - struct of reduced feature vectors
%      idx   - indices of selected features
%      feat_score - score corresponding to idx

% bb idx.first.fhg.de 07/2004


if length(varargin)==1 & ischar(varargin{1}),
  opt= struct('policy', varargin{1}),
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'policy', 'perc_of_weights', ...
                  'select_channels', 0);

if ~exist('model','var') | isempty(model),
  model= struct('classy','LPM', 'msDepth',2, 'std_factor',2);
  model.param= struct('index',2, 'scale','log', 'value', [-2:2:5]);
end
classy= select_model(fv, model, opt);

sz= size(fv.x);
nFeats= prod(sz(1:end-1));

C= trainClassifier(proc_flaten(fv), classy);
if ~isfield(C, 'w'),
  error('classifier must return a field ''w''');
end
  
w= C.w(:);
if length(w)~=nFeats,
  error('classifier weights do not match feature dimension');
end

if opt.select_channels,
  if length(sz)~=3, error('haeh?'); end
  w= reshape(w, sz(1:2));
  w= sqrt(sum(w.^2, 1));
  nFeats= sz(2);
end

[so,si]= sort(-abs(w));
so= -so;
switch(opt.policy),
 case 'number_of_features',
  idx= si(1:retain);
 case 'perc_of_features',
  idx= si(1:ceil(retain/100*nFeats));
 case 'perc_of_weights',
  perc= 100*cumsum(so/sum(so));
  nSel= min([find(perc>retain); nFeats]);
  idx= si(1:nSel);
 otherwise,
  error('policy unknown');
end

if opt.select_channels,
  fv= proc_selectChannels(fv, idx);
else
  if isfield(fv, 'clab'),
    fv= rmfield(fv, 'clab');
  end
  fv= proc_flaten(fv);
  fv.x= fv.x(idx,:);
end

feat_score= so(1:nFeats);
