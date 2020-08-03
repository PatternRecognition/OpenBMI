function out= bbci_calibrate_evalFeature(signal, bbci_feature, mrk)
%BBCI_CALIBRATE_EVALFEATURE - Perform feature extration
%
%Synopsis:
%  FV= bbci_calibrate_evalFeature(EPO, BBCI_FEATURE)
%  FV= bbci_calibrate_evalFeature(CNT, BBCI_FEATURE, MRK)
%
%Arguments:
%  EPO - Structure of epoched data.
%  BBCI_FEATURE - Struct (array) specifying the feature extraction,
%      subfield of 'bbci' structure of bbci_apply.
%  CNT - Structure of continuous data.
%  MRK - Structure of markers.
%
%Output:
%  FV - Structure holding the extracted feature vector

% 11-2011 Benjamin Blankertz


bbci_feature= transformProc2FcnParam(bbci_feature);
bbci_feature= set_defaults(bbci_feature, 'signal', 1);

feature= repmat(struct, [1 length(bbci_feature)]);
for f= 1:length(bbci_feature),
  BF= bbci_feature(f);
  fv= signal(BF.signal);
  if nargin>=3,
    fv= cntToEpo(fv, mrk, BF.ival);
  end

  if f==1 && isfield(fv, 'y'),
    out.y= fv.y;
  end
  
  for k= 1:length(BF.fcn),
    fv= BF.fcn{k}(fv, BF.param{k}{:});
  end
  % clash all feature dimensions (but preserve sample dimension)
  sz= size(fv.x);
  feature(f).x= reshape(fv.x, [prod(sz(1:end-1)) sz(end)]);
end
out.x= cat(1, feature.x);
