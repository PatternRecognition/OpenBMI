function [handles] = val_regression_xy(label,pred,varargin)
% val_regression_xy - X-Y-plot to visualize regression results
%
% Synopsis:
%   val_regression_xy(label,pred)
%   handles = val_regression_xy(label,pred,'Property',Value,...)
%   
% Arguments:
%  label: [1 N] matrix. Target values for regression task
%  pred: [1 N rep] matrix. Model output for the predictive mean for
%      N points. With rep>1, this output is generated from 'rep' repetitions
%      of cross-validation.
%   
% Returns:
%  handles: Struct array with all plot handles. Fields
%     'xy': [G 1] matrix, handles for the x-y-plots for each of the G groups
%     'bound': [G 1] cell, handles for the upper/lower bound plots
%     'diag': handle for the diagonal x=y line
%     'abs': [A 2] matrix, handles for the lines indicating absolute error
%   
% Properties:
%  'marker': String, cell string or cell of cells. Marker for
%     x-y-plot. Everything passed here will be additional arguments for the
%     plot command. Cell of cells as input is only accepted if combined with
%     the <group> option. In this case, each cell entry describes the
%     marker for one group of points.
%  'group': [G 1] cell. Mark points with given index according to
%     <marker>{i}. Default: empty
%  'diagonal': String or cell string. Properties for the diagonal line
%     that indicates perfect prediction. Default: {'r-', 'LineWidth', 2}
%  'abs': [A 1] matrix. Plot additional diagonal lines that indicate the
%     predictions where the absolute error is smaller than the value
%     given here. Eg., using 'abs', [1 2.5] will plot lines indicating
%     regions with absolute error smaller than 1 and smaller than
%     2.5. Default: []
%  'absmarker': [A 1] cell array. Line style for highlighting the i.th
%     region defined with the <abs> option. Default: {}
%  'xjitter': Scalar. Add +- rand(1)*xjitter to the x-position of plotted
%     points (that is, to the label value). Use that to disambiguate plots
%     where many target values are exactly equal. When using jitter,
%     predictions for different crossvalidation runs will be plotted at
%     different x-positions.
%  'upperbound': [1 N rep] matrix. Upper bound or quantile for the
%     prediction. Use only combined with <lowerbound>
%  'lowerbound': [1 N rep] matrix. Lower bound or quantile for the
%     prediction. When both <lowerbound> and <upperbound> are given,
%     vertical lines that indicate upper and lower bound will be plotted
%     with the linestyle given by <boundmarker>
%  'boundmarker': String, cell string, or cell of cells. Properties for
%     the lines that indicate the upper and lower bounds on the
%     prediction. Cell of cells can be used with the <groups> option to
%     have different boundmarkers for groups of points. Default: 'k-'
%  'axis': [a1 a2 a3 a4] matrix, input for 'axis' command
%
% Description:
%   Plot true value (label) on the x-axis versus predicted value (pred)
%   on the y-axis. Optimal prediction would result in a diagonal line, this
%   diagonal is plotted in style <diagonal>. Additional lines that indicate
%   regions where prediction is within +- <abs> are plotted lines in style
%   <absmarker>
%   Additionally, error bars or quantiles on the prediction can be
%   plotted by using the <upperbound> and <lowerbound> options.
%   
% Examples:
%   x-y-plot, show the regions where the prediction error is smaller than
%   1 and where the error is smaller than 2:
%     val_regression_xy(label,pred,'abs', [1 2], ...
%                     'absmarker', {{'k-','LineWidth',2},{'k--'}})
%   Show plot with two groups of points marked differently:
%     val_regression_xy(label,pred,'marker', {{'k.'},{'b.'}},...
%                     'group', {[1:20],[21:size(label,2)]});
%   Plot points with +-1 std error bars:
%     upper = pred+pred_std;
%     lower = pred-pred_std;
%     val_regression_xy(label,pred,'upperbound', upper, ...
%                  'lowerbound', lower, 'boundmarker', 'k-');
%
% See also: 
%   plot
% 

% Author(s), Copyright: Anton Schwaighofer, Sep 2007
% $Id: val_regression_xy.m,v 1.1 2007/09/25 14:40:56 neuro_toolbox Exp $

error(nargchk(2, inf,nargin))
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, 'marker', {'k.'}, ...
                        'group', {}, ...
                        'diagonal', {'r-', 'LineWidth', 2}, ...
                        'abs', [], ...
                        'absmarker', {}, ...
                        'upperbound', [], ...
                        'lowerbound', [], ...
                        'boundmarker', [], ...
                        'axis', [], ...
                        'xjitter', []);
if ischar(opt.marker),
  opt.marker = {{opt.marker}};
end
% marker is of the form {'k.', 'MarkerSize', 6}, and group is
% empty: It seems that the user has just forgotten the extra "outer
% cell", add that
if iscell(opt.marker) & ~iscell(opt.marker{1}) & isempty(opt.group),
  opt.marker = {opt.marker};
end
if iscell(opt.boundmarker) & ~iscell(opt.boundmarker{1}) & isempty(opt.group),
  opt.boundmarker = {opt.boundmarker};
end
if ischar(opt.diagonal),
  opt.diagonal = {opt.diagonal};
end
if ischar(opt.absmarker),
  opt.absmarker = {{opt.absmarker}};
end
if ischar(opt.boundmarker),
  opt.boundmarker = {{opt.boundmarker}};
end

[dim,N,rep] = size(pred);
if size(label,1)~=1 | size(label,2)~=N,
  error('Size of inputs ''label'' and ''pred'' does not match');
end
% Default grouping: all in one
if isempty(opt.group),
  opt.group = {1:N};
end
if xor(isempty(opt.lowerbound), isempty(opt.upperbound)),
  error('To plot bounds, both <lowerbound> and <upperbound> must be specified');
end
if ~isempty(opt.upperbound) & (ndims(opt.upperbound)~=ndims(pred) | ...
                             ~all(size(opt.upperbound)==size(pred))),
  error('Value in option <upperbound> must have same size as predictive mean');
end
if ~isempty(opt.lowerbound) & (ndims(opt.lowerbound)~=ndims(pred) | ...
                             ~all(size(opt.lowerbound)==size(pred))),
  error('Value in option <lowerbound> must have same size as predictive mean');
end
plotbounds = ~isempty(opt.lowerbound) & ~isempty(opt.upperbound);

if length(opt.marker) ~= length(opt.group),
  error('Length of cells in properties <marker> and <group> must match');
end
if length(opt.abs) ~= length(opt.absmarker),
  error('Length of vector in <abs> and cell in <absmarker> must match');
end
if plotbounds & (length(opt.boundmarker) ~= length(opt.group)),
  error('Length of cells in properties <boundmarker> and <group> must match');
end

handles = struct('xy', [], 'bound', [], 'diag', [], 'abs', []);
handles.xy = NaN*ones(size(opt.group));
handles.bound = cell(size(opt.group));
handles.abs = NaN*ones([length(opt.abs) 2]);
for i = 1:length(opt.group),
  % Label, prediction and error bar for the current group of points
  label_i = label(:,opt.group{i});
  pred_i = pred(:,opt.group{i},:);
  % Plot at x position with a given random jitter
  xpos = repmat(label_i, [1 1 rep]);
  xpos = xpos(:);
  if ~isempty(opt.xjitter),
    xjitter = (rand(size(xpos))*2-1)*opt.xjitter;
    xpos = xpos+xjitter;
  end
  % Plot error bars first in the "background"
  if plotbounds,
    % Extent of the error bars
    errmin = opt.lowerbound(:,opt.group{i},:);
    errmax = opt.upperbound(:,opt.group{i},:);
    handles.bound{i} = plot([xpos xpos]', [errmin(:) errmax(:)]', opt.boundmarker{i}{:});
    hold on;
  end
  if i>1,
    hold on;
  end
  handles.xy(i) = plot(xpos, pred_i(:), opt.marker{i}{:});
end
if ~isempty(opt.axis),
  a = opt.axis;
else
  a = axis;
end
a1 = min(a(1), a(3));
a2 = max(a(2), a(4));
if ~isempty(opt.diagonal),
  hold on;
  handles.diag = plot([a1 a2], [a1 a2], opt.diagonal{:});
end
if ~isempty(opt.abs),
  % Mark points that exceed a certain threshold of absolute residuals:
  for i = 1:length(opt.abs),
    v = opt.abs(i);
    handles.abs(i,1) = plot([a1 a2], [a1-v a2-v], opt.absmarker{i}{:});
    handles.abs(i,2) = plot([a1 a2], [a1+v a2+v], opt.absmarker{i}{:});
  end
end
axis(a);
xlabel('True value');
ylabel('Predicted value');
if nargout==0,
  clear handles;
end
