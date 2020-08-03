function [mrk, idx]= mrk_selectTargetDist(mrk, varargin)
%MRK_SELECTTARGETDIST - Exclude close occurences of target stimuli
%
%If the input marker strucutre MRK has a field 'trial_idx', the
%trial structure is respected (i.e., the first event of one trial is
%not regarded as successor of the last event of the previous trial).
%This function assumes that the marker structure is complete (no
%events rejected so far) and chrnologically sorted.
%
%Synopsis:
% [MRK, IDX]= mrk_selectTargetDist(MRK, TARGET_DIST, <OPT>)
% [MRK, IDX]= mrk_selectTargetDist(MRK, <OPT>)
%
%Arguments:
% MRK - Marker structure
% TARGET_DIST - Pair [PRED_DIST SUCC_DIST] specifying how many predecessor
%   and how many successor events should be nontargets. If TARGET_DIST is
%   a scalar, this value is taken for both, predecessor and successor
%   constraint.
% OPT - Struct or property/value list of optional properties:
%   'target_dist': as TARGET_DIST
%   'nontarget_dist': 
%   'verbose': default 0.
%
%Returns:
% MRK - updated marker structre
% IDX - indices of selected markers

% benjamin.blankertz@tu-berlin.de, Jun-2011


if length(varargin)==1 && isnumeric(varargin{1}),
  opt= struct('target_dist', varargin{1});
else
  opt= propertylist2struct(varargin{:});
end
opt= set_defaults(opt, ...
                  'target_dist', 2, ...
                  'nontarget_dist', [], ...
                  'verbose', 0);

if length(opt.target_dist)==1,
  opt.target_dist= [1 1]*opt.target_dist;
end
if isempty(opt.nontarget_dist),
  opt.nontarget_dist= opt.target_dist;
elseif length(opt.nontarget_dist)==1,
  opt.nontarget_dist= [1 1]*opt.nontarget_dist;
end
required_dist= [opt.target_dist; opt.nontarget_dist];

if ~strcmp(mrk.className{1},'target'),
  bbci_warning('assuming class 1 is target, but conflict with mrk.className');
end

if isfield(mrk, 'trial_idx'),
  trial_idx= mrk.trial_idx;
else
  bbci_warning('no field trial_idx: ignoring trial structure');
  trial_idx= 1:length(mrk.pos);
end

valid= zeros(1, length(mrk.pos));
istarget=mrk.y(1,:);
for mm= 1:length(mrk.pos),
  cl= 2-istarget(mm);
  idxT= find(istarget & trial_idx==trial_idx(mm));
  idx_pred= idxT(idxT<mm);
  idx_succ= idxT(idxT>mm);
  dist_pred= min([mm-idx_pred, inf]);
  dist_succ= min([idx_succ-mm, inf]);
  valid(mm)= dist_pred > required_dist(cl,1) && ...
             dist_succ > required_dist(cl,2);
end

idx= find(valid);
mrk= mrk_chooseEvents(mrk, idx);
