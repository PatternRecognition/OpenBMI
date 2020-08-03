function demo_reject()
% DEMO_REJECT - Demo program for classifiers with reject option
%
%   TRAIN_RDAREJECT implements an RDA classifier that can reject test points
%   if their log-likelihood is below a certain threshold. DEMO_REJECT
%   illustrates such a classifier on a small 2-D data set.
%   
%   See also TRAIN_RDAREJECT,APPLY_RDAREJECT
%

% Copyright Fraunhofer FIRST.IDA (2004)
% $Id: demo_reject.m,v 1.4 2004/10/27 16:34:55 neuro_toolbox Exp $

% Generate data from two Gaussian classes with different mean and
% covariance
randn('state', 2);
N1 = 50;
N2 = 20;
mean1 = [0; 0];
mean2 = [2; 2];
X1 = diag([1 0.5])*randn([2 N1]) + repmat(mean1, [1 N1]);
X2 = diag([0.5 1])*randn([2 N2]) + repmat(mean2, [1 N2]);
% Make up the data into the required feature vector format
fv.x = [X1 X2];
fv.y = [repmat([1; 0], [1 N1]) repmat([0; 1], [1 N2])];

h = [];
figure;
h(1) = plot(X1(1,:), X1(2,:), 'bx');
hold on;
h(2) = plot(X2(1,:), X2(2,:), 'bo');
axis([-5 7 -5 7]);
% Train QDA
C = train_RDAreject(fv.x, fv.y, 'lambda', 0, 'gamma', 1, ...
                    'rejectMethod', 'threshold', 'rejectParam', -Inf);
% Evaluate on a large grid
plotboundary(C, 'RDAreject', 1);
caxis([-20 20]);
colorbar;
h2 = plotboundary(C, 'RDAreject');
h(3:4) = h2;
C.rejectParam = -4;
h(5) = plotboundary(C, 'RDAreject');
legend(h, {'Class 1', 'Class 2', 'Decision boundary', 'out==+-1', 'Reject region'});
title('RDA classifier with reject capability');

% Now use a classifier with output scaling: outliers are marked with a value
% close to zero. Also, this classifier has a smooth transition between
% classes (continuous classifier output when crossing the decision boundary)
figure;
axis([-5 7 -5 7]);
hold on;
% Train QDA. Instead of rejecting, scale classifier output by the class
% conditional density of the winning class. Scale such that 95% of the
% probability mass is covered by the classifier, points on the boundary
% of the 95% regions are scaled such that they are scaled at 0.01
scaling = 0.01;
C = train_RDAreject(fv.x, fv.y, 'lambda', 0, 'gamma', 1, 'rejectMethod', ...
                    'classCond', 'rejectParam', [0.95 scaling]);
% Test effect of modifying class probabilities:
% $$$ C.classProb = [0.1 0.9];
% Evaluate on a large grid
[h, x11, x22, x1x2out] = plotboundary(C, 'RDAreject', 1);
plot(X1(1,:), X1(2,:), 'bx');
hold on;
plot(X2(1,:), X2(2,:), 'bo');
% Plot the contour where the classifier output is below the given
% threshold of 0.001*(original classifier output)
contour(x11, x22, x1x2out, [max(max(x1x2out))*scaling min(min(x1x2out))*scaling], 'b-');
legend({'Decision boundary', 'Class 1', 'Class 2', 'Reject region'});
title('RDA classifier with output scaling: Color indicates classifier output');

% Evaluate error rate, to test xvalidation with rejects:
fprintf('Xvalidated negative bitrates, for RDA classifier:\n');
classifier = {'RDAreject', 'lambda', 1, 'gamma', 0, 'rejectBelow', -5};
rand('state', 0);
[loss, dummy, out] = xvalidation(fv, classifier, 'loss', 'negBitrate');
if isstruct(out),
  fprintf('Fraction of rejected samples in each trial:\n');
  1-mean(out.valid, 1)
end

% $$$ fprintf('Result with LDA:\n');
% $$$ rand('state', 0);
% $$$ xvalidation(fv, 'LDA', 'loss', 'negBitrate');


function [h, x11, x22, x1x2out] = plotboundary(C, model, showOutput)
if nargin<3,
  showOutput = 0;
end
a = axis;
nbPoints = 100;
[x11, x22] = meshgrid(a(1):(a(2)-a(1))/nbPoints:a(2),...
                      a(3):(a(4)-a(3))/nbPoints:a(4));
applyFun = getApplyFuncName(model);
out = feval(applyFun, C, [x11(:) x22(:)]');
if ~showOutput,
  rejects = isnan(out);
  if any(rejects),
    out(rejects) = 173;
    x1x2out = reshape(out, [size(x11, 2) size(x22,2)]);
    [dummy, h] = contour(x11, x22, x1x2out, [173 173], 'r-');
    h = h(1);
  else
    x1x2out = reshape(out, [size(x11, 2) size(x22,2)]);
    [dummy, h2] = contour(x11, x22, x1x2out, [0 0], 'k-');
    h(1) = h2(1);
    [dummy, h2] = contour(x11, x22, x1x2out, [-1 1], 'b-');
    h(2) = h2(1);
  end
else
  x1x2out = reshape(out, [size(x11, 2) size(x22,2)]);
  h = pcolor(x11, x22, x1x2out);
  set(h, 'LineStyle', 'none');
  caxis([-4 4]);
  colormap(green_white_red);
  colorbar;
  contour(x11, x22, x1x2out, [0 0], 'k-');
end
