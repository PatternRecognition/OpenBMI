function [epo, iArte]= proc_rejectArtifactsMaxMin(epo, threshold, varargin)
%PROC_REJECTARTIFACTSMAXMIN - Reject epochs according to max-min criterion
%
%This function rejects epochs within which the difference of maximum
%minus minum value exceeds a given threshold. Optionally, the criterion
%can be evaluated on a subset of channels and on a subinterval.
%
%Synopsis:
%  [EPO, IARTE]= proc_rejectArtifactsMaxMin(EPO, THRESHOLD, <OPT>)
%
%Arguments:
%  EPO - Data structure of epoched data, see CntToEpo
%  THRESHOLD - Threshold that evokes rejection, when the difference of
%     maximum minus minimum value within an epoch exceeds this value.
%     for THRESHOLD=0 the function returns without rejection.
%  OPT - Property/value list or struct of optional properties:
%    'clab': [Cell array of Strings]: Channels on which the criterion
%            is evaluated
%    'ival': [START END]: Time interval within the epoch on which the
%            criterion is evaluated
%    'verbose': Sets the level of verbosity. When true, an output informs
%            about the number of rejected trials (if any)
%
%Returns:
%  EPO - Data structure where artifact epochs have been eliminited.
%  IARTE - Indices of rejected epochs

% 05-2011 Benjamin Blankertz


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'clab', '*', ...
                  'ival', [], ...
                  'all_channels', 0, ...
                  'verbose', 0);

if threshold<=0,
  iArte= [];
  return;
end

epo_crit= epo;
if ~isempty(opt.ival),
  epo_crit= proc_selectIval(epo_crit, opt.ival);
end

if ~isempty(opt.clab) && ~isequal(opt.clab,'*'),
  epo_crit= proc_selectChannels(epo_crit, opt.clab);
end

sz= size(epo_crit.x);
epo_crit.x= reshape(epo_crit.x, [sz(1) sz(2)*sz(3)]);
% determine max/min for each epoch and channel:
mmax= max(epo_crit.x, [], 1);
mmin= min(epo_crit.x, [], 1);
% determine the maximum difference (max-min) across channels
if opt.all_channels,
  dmaxmin= min(reshape(mmax-mmin, sz(2:3)), [], 1);
else
  dmaxmin= max(reshape(mmax-mmin, sz(2:3)), [], 1);
end
iArte= find(dmaxmin > threshold);

if opt.verbose,
  fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
          length(iArte), threshold);
end
epo= proc_selectEpochs(epo, 'not',iArte);
