function loss= loss_rocArea(label, out, varargin)
%loss= loss_rocArea(label, out)
%loss= loss_rocArea(label, out, 'Param', Value, ...)
%
% loss defined as area over the ROC curve.
% only for 2-class problems. class 1 is the one to be detected,
% i.e., TP are samples of class 1 classified as '1' (out<0),
% while FP are samples of class 2 classified as '1'.
%
% IN  label - vector of true class labels (1...nClasses)
%     out   - vector of classifier outputs (neg vs. pos)
%
% OUT loss  - loss value (area over roc curve)
%
% Properties:
% 'ignoreNaN': binary. If true, classifier outputs that are NaN will be
%     ignored in the computation of the ROC curve. Default: false (0).
%     In the default setting, NaN effectively count against the classifier.
%
% SEE roc_curve

% Modifications for ignoring NaNs by Anton Schwaighofer
% $Id$

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'ignoreNaN', 0);

if size(label,1)~=2,
  error('roc works only for 2-class problems');
end
if opt.ignoreNaN,
  valid = all(~isnan(out),1);
  label = label(:,valid);
  out = out(valid);
end
N= sum(label, 2);
lind= label2ind(label);

%%resort the samples such that class 2 comes first.
%%this makes ties count against the classifier, otherwise
%%loss_rocArea(y, ones(size(y))) could result in a loss<1.
[so,si]= sort(-lind);
lind= lind(:,si);
out= out(:,si);

[so,si]= sort(out);
lo= lind(si);
idx2= find(lo==2);
ncl1= cumsum(lo==1);
roc= ncl1(idx2)/N(1);

%% area over the roc curve
loss= 1 - sum(roc)/N(2);
