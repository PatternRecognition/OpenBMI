function feature= bbci_apply_evalFeature(signal, bbci_feature, event)
%BBCI_APPLY_EVALFEATURE - Perform feature extration
%
%Synopsis:
%  FEATURE= bbci_apply_evalFeature(SIGNAL, BBCI_FEATURE, EVENT)
%
%Arguments:
%  SIGNAL - Structure buffering the continuous signals,
%      subfield of 'data' structure of bbci_apply, see bbci_apply_evalSignal
%  BBCI_FEATURE - Structure specifying the feature extraction,
%      subfield of 'bbci' structure of bbci_apply.
%  EVENTS - Array of Struct, specifying the event(s) at which a control
%      signal should be calculated. Each Struct has the fields
%      'time' and 'desc' like in the marker context, 
%      see bbci_apply_evalCondition.
%
%Output:
%  FEATURE - Structure holding the extracted feature vector

% 02-2011 Benjamin Blankertz


fv= bbci_apply_getSegment(signal, event.time, bbci_feature.ival);

for k= 1:length(bbci_feature.fcn),
  fv= bbci_feature.fcn{k}(fv, bbci_feature.param{k}{:});
end
feature.x= fv.x(:);
feature.time= event.time;
