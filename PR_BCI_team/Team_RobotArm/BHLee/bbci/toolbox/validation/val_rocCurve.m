function [roc_y, auc, roc_x, hp, threshold]= val_rocCurve(label, out, varargin)
%[roc, auc, roc_x, hp,threshold]= val_rocCurve(label, out, <opt>)
%
% ROC - receiver operator characteristics
% Only defined for 2-class problems. 
% By default, class 1 is the one to be detected,
% i.e., TP are samples of class 1 classified as '1' (out<0),
% while FP are samples of class 2 classified as '1'.
% This can be reversed by opt.detect_class.
%
% Note that ties (two samples are mapped to the same classifier output)
% are always counted against the classifier.
%
% IN   label  - true class labels, can also be a data structure (like epo)
%               including label field '.y'
%      out    - classifier output (as given, e.g., by the third output
%               argument of xvalidation)
%               size is [1 nSamples nShuffles]. when nShuffles is >1,
%               the resulting curve is the average of the ROC curves
%               of each shuffle.
%      opt 
%      .detect_class - index of class to be detected, 1 (default) or 2.
%      .plot      - plot ROC curve, default when no output argument is given.
%      .linestyle - cellarray of linestyle property pairs,
%                   default {'linewith',2}.
%      .xlabel    - label of the x-axis, default 'false positive rate'.
%      .ylabel    - label of the y-axis, default 'true positive rate'.
%      .ignoreNaN - binary. If true, classifier outputs that are NaN in
%                   any of the repetitions of the xvalidation will
%                   be ignored in the computation of the ROC curve. 
%                   Default: false (0). In the default setting, NaNs
%                   effectively count against the classifier. 
%
% OUT  roc   - ROC curve
%      auc   - area under the ROC curve
%      roc_x - x-axis for plotting the ROC curve
%      hp    - handle to the plot
%      threshold - Threshold on the classifier output for each point of
%          the ROC curve. If nShuffles>1, this is averaged over all
%          shuffles.
% Remark:
%   The x-axis of the ROC curve plots the probability of false alarms,
%     p (classified as Positive | belongs to class Negative )
%     = FP/(FP+TN) = FP/N(2), where N(2) is the number of negatives
%   The x-axis is here labelled with 'false positive rate', yet the term
%   'false positive rate' is sometimes also used for the quantity
%     p (belongs to class Negative | classified as Positive )
%   The y-axis of the ROC curve plots the true positive rate,
%     = TP/(TP+FN) = TP/N(1), where N(1) is the number of positives
%
%
% SEE  xvalidation

% bb ida.first.fhg.de
% Modifications for ignoring NaNs and logging thresholds by Anton Schwaighofer
% $Id:$

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'plot', nargout==0, ...
                  'xlabel', 'false positive rate', ...
                  'ylabel', 'true positive rate', ...
                  'linestyle', {'linewidth',2}, ...
                  'detect_class', 1, ...
                  'ignoreNaN', 0);

if isstruct(label),
  label= label.y;
end
if size(label,1)>2,
  error('roc works only for 2-class problems');
end
if size(out,1)~=1,
  error('??? first dimension of out should be singleton');
end
if opt.detect_class==2,
  % Just flip labels and classifier output, if class 2 is to be detected
  label = flipud(label);
  out = -out;
elseif opt.detect_class~=1,
  error('opt.detect must be 1 or 2');
end

if opt.ignoreNaN,
  % Remove every point that is NaN in any of the repetitions. This
  % removes too much (a rejected point in one repetition kicks out this
  % point in the full evaluation). Yet, otherwise, we could not do
  % computation of ROC curves per repetition and subsequent averaging
  valid = all(~isnan(out),3);
  label = label(:,valid);
  out = out(1,valid,:);
end
N= sum(label, 2);


%%resort the samples such that class 2 comes first.
%%this makes ties count against the classifier, otherwise
%%val_rocCurve(y, ones(1,size(y,2))) could result in an auc>0.
% "Ties" means here that two or more samples are mapped to the same
% classifier output, eg. two positive and one negatives have the same
% classifier output. It is unclear how this should be resolved, thus we
% are pessimistic and put in the one negative example first.
[so,si]= sort([1 -1]*label);
label= label(:,si);
out= out(1,si,:);

% x-axis of the roc curve: Possible values for the FP rate are only 
% 0, 1/N(2), 2/N(2), ... 1, where N(2) is the number of 'negatives' (size
% of the neutral class)
roc_x= linspace(0, 1, N(2)+1);
roc_x= roc_x(floor(1:0.5:N(2)+1.5));

nShuffles= size(out,3);
ROC= zeros(nShuffles, N(2));
Threshold= zeros(nShuffles, N(2));
for ii= 1:nShuffles,
  % Sort classifier output
  [so,si]= sort(out(1,:,ii), 2);
  lo= label(:,si);
  % Find those positions where we get another 'negative' while increasing
  % the threshold (FP rate will change)
  idx2= find(lo(2,:));
  % Cumulative sum of 'positives'
  ncl1= cumsum(lo(1,:));
  % ROC curves is given by TP rate on those positions where the FP rate
  % has changed
  ROC(ii,:)= ncl1(idx2)/N(1);
  Threshold(ii,:) = so(idx2);
end
roc= mean(ROC, 1);
threshold = mean(Threshold, 1);

%% area under the roc curve
auc= sum(roc)/N(2);

%% make it a step function (to please StH)
roc_y= [0 roc(floor(1:0.5:N(2)+0.5)) 1];
threshold = [-Inf threshold(floor(1:0.5:N(2)+0.5)) Inf];

if opt.plot,
  hp= plot(roc_x, roc_y, opt.linestyle{:});
  xlabel(opt.xlabel);
  ylabel(opt.ylabel);
  title(sprintf('area under curve= %.4f', auc));
  axis([-0.05 1.05 -0.05 1.05], 'square');
else
  % Make sure that the return argument is defined even if no plotting was done
  hp = [];
end

% Clear the variable if no output arguments requested. Otherwise, calling
% without trailing semicolon would print out the contents of roc_y
if nargout==0,
  clear roc_y;
end
