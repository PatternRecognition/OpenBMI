function [prc_x, prc_y, hp, threshold]= val_precisionRecallCurve(label, out, varargin)
% val_precisionRecallCurve - Compute and plot precision/recall curve
%
% Synopsis:
%   val_precisionRecallCurve(label,out)
%   [prc_x,prc_y,hp,threshold] = val_precisionRecallCurve(label,out)
%   [prc_x,prc_y,hp,threshold] = val_precisionRecallCurve(label,out,'Property',Value,...)
%   
% Arguments:
%  label: [2 N] or [1 N] matrix, class membership
%  out: [1 N nShuffles] matrix. Classifier output for N examples in
%      nShuffles repetitions of xvalidation
%   
% Returns:
%  prc_x: [1 nClass1+1] matrix. Precision for different values of
%      classifier threshold
%  prc_y: [1 nClass1+1] matrix. Recall for different values of classifier
%      threshold
%  hp: Vector of plot handles (precision recall curves and patch for
%      range spanned)
%  threshold: [1 nClass1+1] matrix, classifier threshold to achieve a
%      certain precision/recall
%   
% Properties:
%  average: String, one of {'mean','shading','none'}. How should results
%      over xvalidaton shuffles be averaged? 'mean': Average precision
%      achieved for given recall. Can be misleading! 'shading': As 'mean',
%      but also plots a patch that shows the range spanned over all
%      xvalidation shuffles. 'none': Plots nShuffles precision/recall
%      curves. Default: 'shading'
%  detect_class: 1 or 2. Class to be detected, i.e., which class is the
%      'positive' class? Default: 1
%  ignoreNaN: binary. If true, classifier outputs that are NaN in any of
%      the shuffles will be ignored in the computation of the ROC
%      curve. Default: false (0). In the default setting, NaNs effectively
%      count against the classifier.
%  linestyle: cellarray of linestyle property pairs. Default:
%      {'linewidth',2}
%  shading: [1 3] vector, RGB-triplet for the shading of the precision
%      range plot
%  xlabel: String for x-axis. Default: 'Precision TP/(TP+FP)'
%  ylabel: String for y-axis. Default: 'Recall TP/P'
%  plot: If false, only compute precision/recall curves, do not
%      plot. Default: nargout==0
%   
% Description:
%   Precision/recall curves are tools to evaluate the quality of a
%   classifier in terms of the ranking produced.
%     Precision = TP/(TP+FP) = TP/(number of positively classified examples)
%     Recall = TP/P = TP/(true number of positives)
%   In contrast to ROC curves, precision/recall curves are highly
%   sensitive to the class balance. P/R curves must not be monotonous.
%   
%   P/R curves start at recall 1/nClass1 (only 1 example is classified as
%   positive). Depending on the true label, the corresponding precision is
%   either 0 or 1. The P/R curve ends at recall 1, with a corresponding
%   precision that is equal to the class prior nClass1/(nClass1+nClass2).
%   The P/R curves are computed as 'minimum precision that is achieved for a
%   given recall'.
%
%   If results for several xvalidation runs are given and option <average>
%   is set to 'shading', the average precision for given recall is
%   plotted, with a patch that indicates the minimum and maximum
%   precision achieved in each xvalidation run.
%
%   
% Examples:
%   A classifier that puts a 'negative' at the top position of the
%   ranking: recall 0 is a defined point
%     val_precisionRecallCurve([1 1 2 2], [-1 -1 -2 1])
%   Plot P/R curve & highlight the range spanned over different shuffles:
%     val_precisionRecallCurve([1 1 2 2], reshape([-1 -1 -2 1; -1 -1 0 0],[1 4 2]))
%   P/R curves are computed 'pessimistically': In this example, for a
%   recall of 1, we can achieve a precision of either 1, 1/2 or
%   1/3. Returned value is 1/3 in this case
%     val_precisionRecallCurve([1 1 1 2 2 2 2], [-1 1 1.5 0 0 2 3])
%   
% See also: val_rocCurve
% 

% Author(s), Copyright: Anton Schwaighofer, Apr 2006
% $Id: val_precisionRecallCurve.m,v 1.3 2006/04/16 14:50:27 neuro_toolbox Exp $


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'plot', nargout==0, ...
                  'xlabel', 'Precision TP/(TP+FP)', ...
                  'ylabel', 'Recall TP/P', ...
                  'linestyle', {'linewidth',2}, ...
                  'shading', [0.8 1 1], ...
                  'average', 'shading', ...
                  'detect_class', 1, ...
                  'ignoreNaN', 0);

if isstruct(label),
  label= label.y;
end
if size(label,1)==1,
  label = ind2label(label);
end
if size(label,1)>2,
  error('Precision/recall curves can only be plotted for 2-class problems');
end
if size(out,1)~=1,
  error('First dimension of out should be singleton');
end
if size(label,2)~=size(out,2),
  error('Number of examples must match with size of classifier output');
end
if opt.detect_class==2,
  % Just flip labels and classifier output, if class 2 is to be detected
  label = flipud(label);
  out = -out;
elseif opt.detect_class~=1,
  error('Property <detect> must be 1 or 2');
end

if opt.ignoreNaN,
  % Remove every point that is NaN in any of the repetitions. This removes too
  % much (a rejected point in one repetition kicks out this point in the
  % full evaluation). Yet, otherwise, we could not compute precision/recall
  % curves per repetition with subsequent averaging
  valid = all(~isnan(out),1);
  label = label(:,valid);
  out = out(1,:,valid);
end
N= sum(label, 2);


%%resort the samples such that class 2 comes first.
%%this makes ties count against the classifier
% "Ties" means here that two or more samples are mapped to the same
% classifier output, eg. two positive and one negatives have the same
% classifier output. It is unclear how this should be resolved, thus we
% are pessimistic and put in the one negative example first.
[so,si]= sort([1 -1]*label);
label= label(:,si);
out= out(1,si,:);

% y-axis: Possible values for the recall (=TP rate = TP/P) are only 1,
% (N(1)-1)/N(1), (N(1)-2)/N(1), ... 0, where N(1) is the number of
% 'positives' (size of the class to be detected)
% Result for recall 0 is only defined if the first example in the sorted
% classifier output is a negative
recall = linspace(0, 1, N(1)+1);

nShuffles= size(out,3);
PRC= zeros(nShuffles, N(1)+1);
Threshold= zeros(nShuffles, N(1)+1);
for ii= 1:nShuffles,
  % Sort classifier output
  [so,si]= sort(out(1,:,ii), 2);
  lo= label(:,si);
  % Find those positions where we get another 'positive' while increasing the
  % threshold.
  idx1= find(lo(1,:));
  % Cumulative sum of 'positives'
  ncl1= cumsum(lo(1,:));
  % TP+FP is the number of examples that are classified as positives:
  % We compute the precision at the point just before the recall is changing,
  % this way, we obtain the minimum precision at a given recall rate (yes,
  % we're a pessimistic bunch)
  nClassified1= idx1(2:end)-1;
  % Precision/recall curves is given by minimum precision on those positions
  % where the recall has changed
  PRC(ii,2:(end-1))= ncl1(idx1(2:end)-1)./nClassified1;
  PRC(ii,end) = ncl1(end)./size(out,2);
  Threshold(ii,2:end) = [so(idx1(2:end)-1) Inf];
  if idx1(1)<2,
    % First example of sorted classifier output is positive: precision at
    % recall 0 is undefined
    PRC(ii,1) = NaN;
    Threshold(ii,1) = -Inf;
  else
    % First example is a negative: recall 0, precision 0 while increasing
    % the threshold until we hit the first true positive
    PRC(ii,1) = 0;
    Threshold(ii,1) = so(idx1(1)-1);
  end
end
switch lower(opt.average)
  case 'mean'
    prc_x = mean(PRC, 1);
    threshold = mean(Threshold, 1);
  case {'none', []', 'shading'}
    prc_x = PRC;
    threshold = Threshold;
  otherwise
    error('Property <average> must be one of ''mean'',''none''');
end
prc_y = recall;

if opt.plot,
  switch lower(opt.average)
    case 'shading'
      if nShuffles>1,
        % Create a patch that spans the minimum and maximum precision
        leftx = min(prc_x, [], 1);
        rightx = max(prc_x, [], 1);
        patchy = recall;
        hp(2) = patch([leftx fliplr(rightx)], [patchy fliplr(patchy)], opt.shading);
        set(hp(2), 'LineStyle', 'none');
        hold on;
      end
      hp(1) = plot(mean(PRC,1), prc_y, opt.linestyle{:});
    otherwise  
      for i = 1:size(prc_x,1),
        hp(i) = plot(prc_x(i,:), prc_y, opt.linestyle{:});
        hold on;
      end
  end
  xlabel(opt.xlabel);
  ylabel(opt.ylabel);
  title('Precision/Recall curve');
  axis([-0.05 1.05 -0.05 1.05], 'square');
else
  % Make sure that the return argument is defined even if no plotting was done
  hp = [];
end

% Clear the first output variable if no output arguments
% requested. Otherwise, calling without trailing semicolon would print out
% the contents of prc_x
if nargout==0,
  clear prc_x;
end
