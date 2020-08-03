function [cal_y, cal_x, handles, hist_norm]= val_calibrationCurve(label, out, varargin)
% val_calibrationCurve - Plot calibration curve (evaluate predictive uncertainty)
%
% Synopsis:
%   val_calibrationCurve(label,out)
%   [cal_y,cal_x,handles,hist_norm] = val_calibrationCurve(label,out,Property,Value,...)
%   
% Arguments:
%   label: True class labels, can also be a feature vector data structure with
%       field .y
%   out: [1 nSamples nShuffles] matrix. Classifier output, continuous or
%       probabilities (see below). When nShuffles is >1, average results over shuffles.
%   
% Returns:
%   cal_y: [1 d] vector. Calibration curve, y values
%   cal_x: [1 d] vector. Calibration curve, x values (position of bin centers)
%   handles: struct array of handles for different parts of the plot
%   hist_norm: [1 d] vector. Normalized histogram for each bin of the
%       classifier output
%   
% Properties:
%   'sigmoid': Scalar. If 0, assume that classifier outputs a probability
%       between 0 and 1. If ~=0, assume non-probabilistic classifier
%       output and squash via transfer function
%       1/(1+exp(-(sigmoid*out+bias))).
%       Default value: 1 (assume non-probabilistic classifier output)
%   'bias': Scalar. A value to add to the classifier output before
%       applying the sigmoid transfer function. Default value: 0
%   'plot': If non-zero, plot the curve. Defaults to 1 if no output argument
%       given
%   'bins': Number of bins for quantizing classifier output. Default: 20
%   'edges': Edges of the bins for quantizing classifier output. This
%       overrides the 'bins' option. Default value: []
%   'linestyle': Line style for the calibration curve (property/value
%       list in a cell array). Default: {'LineWidth', 2}
%   'diagonal': If non-zero, plot a diagonal line in style
%       'k--'. Default: 1
%   'attributes': If non-zero, make an attributes plot (see below). Default: 1.
%   'histogram': One of {0, 'none', 'bar', 'plot'}. Plot histograms (with
%       'bar') or crosses (with 'plot') that indicate the fraction of
%       points that lies within a given bin for the classifier
%       output. Default: 0 (no histogram)
%   'histogramstyle': Cell array of property/value pairs for plotting the
%       histogram bars or plots. Default value is chosen based on the
%       <histogram> property
%   'shading': [1 3] vector, RGB-triplet for the shading of the attributes plot
%   'xlabel': Label for x-axis, default: 'predicted confidence f for class 2'
%   'ylabel': Label for y-axis, default: 'empirical confidence p(class 2 | f)'
%   
% Description:
%   Calibration curves are a tool to evaluate the quality of a classifier in
%   giving reasonable predictive variances. Assuming that a classifier
%   outputs a probability f that indicatives the membership in class 2, the
%   calibration curve plots p(x in class 2 | f) versus f.  An optimal
%   calibration curve is a diagonal line going from (0,0) to (1,1).
% 
%   Intention: 'We want a tornado forecast of 10% to mean that exactly
%   10% of such forecasts are in fact tornadoes.' (C. Marzban)
%   
%   In contrast to the ROC curve, this routines needs probabilities as
%   inputs. If your classifier outputs true probabilities, call this routine
%   with 'sigmoid'=0. Alternatively, for 'sigmoid'~=0, take continuous
%   classifier output, scale and put into transfer function
%   1/(1+exp(-(sigmoid*out+bias))).  Bear in mind that the scaling here has big
%   influence on the shape of the curve. Large values make the classifier
%   over-confident.
%
%   With the 'attributes' option, one can also generate the so-called
%   attributes plot. Here, a shaded region is plotted under the
%   calibration curve. Points of the calibration curve that lie in this
%   region have a positive contribution to the Brier score. For these
%   points, non-trivial predictions have been made.
%
%   With the 'histogram' option, it is possible to monitor the range of
%   the classifier output. A normalized histogram is plotted under the
%   calibration curve, that indicates what fraction of points attains a
%   specific value of the classifier output.
%
%   The calibration curve can also be used to adjust the bias of a
%   classifier. For example, try
%     val_calibrationCurve(label, out+bias, 'sigmoid', scale)
%   for different values of bias and scale, until the calibration curve
%   is close to the diagonal.
%
%   Alternatively, use val_calibrationCurveUI for a graphical user
%   interface to tune calibration curve pararameters.
%   
%   
% Examples:
%   For a classifier that outputs probabilities: 
%     val_calibrationCurve(label, out, 'sigmoid', 0)
%   For a classifier that outputs continuous values: Scale by 0.1 and squash
%     val_calibrationCurve(label, out, 'sigmoid', 0.1)
%   Change colors of the plot: Lines thin, shaded area in light yellow:
%     val_calibrationCurve(label, out, 'sigmoid', 0.1, 'linestyle',
%     {'LineWidth', 1}, 'shading', '[1 1 0.8])
%   If the classes 1 and 2 are highly unbalanced, the default value of 20
%   bins to discretize the classifier output may be too high. Try
%     val_calibrationCurve(label, out, 'bins', 10)
%   
%   
% References:
%   Marzban, Caren: Performance Assessment. Unpublished Manuscript. 
%   http://www.nhn.ou.edu/~marzban/marzban_perf.pdf
%   
% See also: val_calibrationCurveUI,val_rocCurve,patch
% 

% Author(s): Anton Schwaighofer, Feb 2005
% $Id: val_calibrationCurve.m,v 1.7 2007/09/24 09:48:49 neuro_toolbox Exp $


opt= propertylist2struct(varargin{:});
noScaleGiven = ~isfield(opt, 'sigmoid');
opt= set_defaults(opt, ...
                  'plot', nargout==0, ...
                  'bins', 20, ...
                  'edges', [], ...
                  'bias', 0, ...
                  'sigmoid', 1, ...
                  'diagonal', 1, ...
                  'histogram', 0, ...
                  'histogramstyle', {}, ...
                  'attributes', 1, ...
                  'shading', [0.8 1 1], ...
                  'xlabel', 'Predicted confidence f for class 2', ...
                  'ylabel', 'Empirical confidence p(class 2 | f)', ...
                  'linestyle', {'linewidth',2});

if size(label,1)>2,
  error('Calibration curves can only be plotted for 2-class problems');
end
if size(out,1)~=1,
  error('First dimension of out should be singleton');
end
if size(label,2)~=size(out,2),
  error('Number of examples must match with size of classifier output');
end
if isempty(opt.histogramstyle),
  switch opt.histogram
    case 'plot'
      opt.histogramstyle = {'k+'};
    case 'bar'
      opt.histogramstyle = {'FaceColor', 0.6*[1 1 1]};
  end
end

% With sigmoid==0, assume that the classifier outputs are already
% probabilies. Otherwise, apply sigmoid squashing function
if opt.sigmoid~=0,
  if noScaleGiven,
    warning('Applying sigmoid function with default scaling. This may result in misleading curves.');
  end
  f = 1./(1+exp(-(opt.sigmoid*out+opt.bias)));
elseif any(out<0 | out>1),
  error('out needs to contain probabilities with the ''sigmoid''==0 option');
else
  f = out;
end
if isstruct(label),
  label= label.y;
end

if isempty(opt.edges),
  % Bin edges, if not given explicitly. histc uses exact matches for the
  % boundary of the last bin, thus make the last bin a bit wider
  opt.edges = linspace(0,1,opt.bins+1);
  opt.edges(end) = 1.01;
end
% Plot the curve on the bin centers
cal_x = opt.edges(1:end-1)+diff(opt.edges)/2;

fClass1 = f(1,logical(label(1,:)),:);
fClass2 = f(1,logical(label(2,:)),:);
% Compute histogram for p(f|x=1) and p(f|x=2). For the case of
% cross-validated output, overage over the individual runs (count
% histogram over intra-class data over all runs)
f1 = histc(fClass1(:), opt.edges)';
f2 = histc(fClass2(:), opt.edges)';
f1 = f1(1:end-1);
f2 = f2(1:end-1);
% Joint probability table
joint = [f1; f2];
% Conditional p(x=2|f)
sumJoint = sum(joint,1);
non0 = sumJoint~=0;
x2_f = joint(2,non0)./sumJoint(non0);

cal_y = x2_f;
cal_x = cal_x(non0);
hist_norm = sumJoint(non0)./sum(sumJoint);

handles = struct('calibration', NaN, 'diagonal', NaN, 'attributes', NaN, ...
                 'histogram', NaN);
if opt.plot,
  % Check for attribute diagram: 
  if opt.attributes,
    classFreq = sum(joint,2);
    c2 = classFreq(2)/sum(classFreq);
    % The bisector (Winkelhalbierende) of the angle made by the diagonal
    % and a horizontal line at y-value c2. The angle is always 45deg
    % (pi/4). Thus, the height above the horizontal is sin(pi/8)*(1-c2)/cos(pi/8)
    h2 = tan(pi/8)*(1-c2);
    h1 = tan(pi/8)*c2;
    % Create patch for those points that contribute positively to the
    % Brier score:
    patchx = [0 c2 c2 0; c2 1 1 c2]'; 
    patchy = [0 0 c2 c2-h1; c2 c2+h2 1 1]';
    handles.attributes = patch(patchx, patchy, opt.shading);
    set(handles.attributes, 'LineStyle', 'none');
    hold on;
  end
  switch opt.histogram
    case {'none', 0}
    case 'plot'
      handles.histogram = plot(cal_x, hist_norm, opt.histogramstyle{:});
      hold on;
    case 'bar'
      handles.histogram = bar(cal_x, hist_norm, 'hist', 0.7);
      set(handles.histogram, opt.histogramstyle{:});
      hold on;
    otherwise
      error('Unknown value for option <histogram>');
  end
  handles.calibration = plot(cal_x, cal_y, opt.linestyle{:});
  xlabel(opt.xlabel);
  ylabel(opt.ylabel);
  title('Calibration Curve');
  if opt.diagonal,
    hold on;
    handles.diagonal = plot([0 1], [0 1], 'k--');
  end
  axis([-0.05 1.05 -0.05 1.05], 'square');
end

% Clear the variable if no output arguments requested. Otherwise, calling
% without trailing semicolon would print out the contents of cal_y
if nargout==0,
  clear cal_y;
end
