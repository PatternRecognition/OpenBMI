function opt= val_calibrationCurveUI(label, out, varargin)
% val_calibrationCurveUI - GUI for choosing optimal parameters for val_calibrationCurve
%
% Synopsis:
%   opt = val_calibrationCurveUI(label,out)
%   
% Arguments:
%  label: True class labels, can also be a feature vector data structure
%      with field .y
%  out: [1 nSamples nShuffles] matrix. Classifier output, continuous or
%      probabilities. When nShuffles is >1, average results over shuffles.
%   
% Returns:
%  opt: Struct with field .bins, .sigmoid, and .bias. The values
%      correspond to those chosen with the respective sliders for number of
%      histogram bins, scaling, and bias, at the time the window was closed.
%   
% Properties:
%  'title': String. A title to be put into the GUI window (figure
%      property 'Name'). Default: ''.
%  Also, all valid properties of val_calibrationCurve can be passed as
%  options here, see val_calibrationCurve for detailed descriptions. In
%  particular, options .sigmoid, .bias, and .bins are used as the initial
%  values for the GUI.
%   
% Description:
%   Usually, classifier outputs are not scaled to give well calibrated
%   systems. This GUI should help determining optimal parameters, in
%   order to get a calibration curve that is close to the diagonal. It
%   allows a easy tuning of scale and bias parameters via sliders, and
%   immediately displays the changed calibration curve.
%   
%   
% Examples:
%   Determine optimal parameters:
%     opt = val_calibrationCurveUI(label, out);
%   Once this is finished, call the actual plot routines with the
%   returned parameters:
%     [cal_y, cal_x] = val_calibrationCurve(label, out, opt);
%   
% See also: val_calibrationCurve
% 

% Author(s), Copyright: Anton Schwaighofer, Apr 2005
% $Id: val_calibrationCurveUI.m,v 1.2 2005/07/04 09:03:10 neuro_toolbox Exp $

global CALIBCURVECHANGED;

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'bins', 20, ...
                  'sigmoid', 1, ...
                  'bias', 0, ...
                  'title', '');

f = figure;
set(f, 'MenuBar','none');
if ~isempty(opt.title),
  set(gcf, 'Name', opt.title);
end

biasT = uicontrol(f, 'Style', 'Text', 'Units', 'normalized', ...
                 'position', [0.2 0.95 0.6 0.04], 'String', 'Classifier bias:',...
                  'FontSize', 12);
bias = uicontrol(f, 'Style', 'Slider', 'Units', 'normalized', ...
                 'position', [0.2 0.9 0.6 0.05], ...
                 'Min', -3, 'Max', 3, 'SliderStep', [0.005 0.1], 'Value', opt.bias, ...
                 'Tooltipstring', 'Classifier bias', ...
                 'Callback',@generalCallback);
sigmoidT = uicontrol(f, 'Style', 'Text', 'Units', 'normalized', ...
                     'position', [0.2 0.85 0.6 0.04], 'String', 'Classifier scaling:',...
                     'FontSize', 12);
sigmoid = uicontrol(f, 'Style', 'Slider', 'Units', 'normalized', ...
                    'position', [0.2 0.8 0.6 0.05], ...
                    'Min', -5, 'Max', 5, 'Value', log(opt.sigmoid), ...
                    'SliderStep', [0.005 0.05], ...
                    'Tooltipstring', 'Classifier scaling', ...
                    'Callback',@generalCallback);
binsT = uicontrol(f, 'Style', 'Text', 'Units', 'normalized', ...
                  'position', [0.2 0.75 0.6 0.04], 'String', 'Number of histogram bins:',...
                  'FontSize', 12);
bins = uicontrol(f, 'Style', 'Slider', 'Units', 'normalized', ...
                 'position', [0.2 0.7 0.6 0.05], ...
                 'Min', 5, 'Max', 50, 'Value', opt.bins, ...
                 'Tooltipstring', 'Number of histogram bins', ...
                 'Callback',@generalCallback);
closeWin = uicontrol(f, 'Style', 'Pushbutton', 'Units', 'normalized', ...
                  'position', [0.75 0.1 0.2 0.07], 'String', 'Close Window',...
                  'FontSize', 12, 'Callback', 'closereq;');
curve = axes('position', [0.2 0.1 0.6 0.6]);

oldBins = get(bins, 'Value');

% We promised to only return the relevant fields, remove the options
% identifier tag
if isfield(opt, 'isPropertyStruct'),
  opt = rmfield(opt, 'isPropertyStruct');
end
opt = rmfield(opt, 'title');

CALIBCURVECHANGED = 1;
% Sigmoid must be the second entry here!
texts = [biasT sigmoidT binsT];
handles = [bias sigmoid bins];

while ismember(f, get(0, 'Children')),
  if CALIBCURVECHANGED,
    opt.bias = get(bias, 'Value');
    opt.sigmoid = exp(get(sigmoid, 'Value'));
    % Make sure we always have an integer value, but it must change at
    % least by one
    newBins = get(bins, 'Value');
    diff = newBins-oldBins;
    set(bins, 'Value', oldBins+sign(diff)*ceil(abs(diff)));
    opt.bins = get(bins, 'Value');
    oldBins = opt.bins;
    % Set the slider label texts to the respective values:
    for i = 1:length(handles),
      h = texts(i);
      s = get(h, 'String');
      colon = findstr(s, ':');
      value = get(handles(i), 'Value');
      % Slider for scaling is on log scale:
      if i==2,
        value = exp(value);
      end
      s = [s(1:colon) ' ' num2str(value)];
      set(h, 'String', s);
    end
    axes(curve);
    cla;
    val_calibrationCurve(label, out, opt);
    CALIBCURVECHANGED = 0;
  end
  pause(0.1);
end


function generalCallback(h, varargin)
global CALIBCURVECHANGED;
CALIBCURVECHANGED=1;

