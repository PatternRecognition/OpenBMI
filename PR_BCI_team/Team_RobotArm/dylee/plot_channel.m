function H= plot_channel(epo, clab, varargin)
%PLOT_CHANNEL - Plot the classwise averages of one channel of 1D (time OR
% frequency) or 2D data (time x frequency)
%
%Synopsis:
% H= plot_channel(EPO, CLAB, <OPT>)
%
%Input:
% EPO:  Struct of epoched signals, see makeEpochs
% CLAB: Name (or index) of the channel to be plotted.
% OPT:  Struct or property/value list of optional
% properties, used to generate a plot of 1D or 2D data:
%  ----------------------------------
%   (1) 1D channel data plot:
%  .Butterfly    - Butterfly plot. Recommended when CLAB =  '*'. Only
%                  first class is considered for plotting. If ColorOrder is
%                  default, the current colormap is used.
%  .PlotStat     - plot additional statistic: 'std' for standard deviation,
%                  'sem' for standard error of the mean, 'perc' for percentiles
%                  or 'none' for nothing.
%  .Legend       - show Class legend (1, default), or not (0).
%  .LegendPos    - position of the legend, see help of function 'legend'.
%  .XUnit        - unit of x axis, default 'ms'
%  .YUnit        - unit of y axis, default epo.yUnit if this field
%                  exists, 'a.u.' otherwise
%  .YDir         - 'normal' (negative down) or 'reverse' (negative up)
%  .RefCol       -  Color of patch indicating the baseline interval
%  .RefVSize     - (value, default 0.05) controls the Height of the patch
%                  marking the Reference interval.
%  .ColorOrder   - specifies the Colors for drawing the curves of the
%                  different Classes. If not given the ColorOrder
%                  of the current axis is taken. As special gimmick
%                  you can use 'rainbow' as ColorOrder.
%  .LineWidthOrder - (numerical vector) analog to ColorOrder.
%  .LineStyleOrder - (cell array of strings) analog to ColorOrder.
%  .LineSpecOrder - (cell array of cell array) analog to ColorOrder,
%                  giving full control on the appearance of the curves.
%                  (If LineSpecOrder is defined, LineWidthOrder and
%                  LineStyleOrder are ignored, but ColorOrder is not.)
%  .XGrid, ...   - many axis properties can be used in the usual
%                  way
%  .YLim         - Define the y limits. If empty (default) an automatic
%                  selection according to 'YLimPolicy' is performed.
%  .YLimPolicy   - policy of how to select the YLim (only if 'YLim' is
%                  empty resp unspecified): 'auto' uses the
%                  usual mechanism of Matlab; 'tightest' selects YLim as the
%                  exact data range; 'tight' (default) takes the data range,
%                  adds a little border and selects 'nice' limit values.
%  .Title        - title of the plot to be displayed above the axis. 
%                  If OPT.title equals 1, the channel label is used.
%  .Title*       - with * in {'Color', 'FontWeight', 'FontSize'}
%                  selects the appearance of the title.
%  .XZeroLine    - draw an axis along the x-axis at y=0
%  .YZeroLine    - draw an axis along the y-axis at x=0
%  .ZeroLine*    - with * in {'Color','Style'} selects the
%                  drawing style of the axes at x=0/y=0
%  .AxisTitle    - (string) title to be displayed *within* the axis.
%  .AxisTitle*   - with * in {'Color', 'HorizontalAlignment',
%                  'VerticalAlignment', 'FontWeight', 'FontSize'}
%                  selects the appearance of the subplot titles.
%  ----------------------------------
%   (2) 2D channel data plot:
%  .XUnit  - unit of x axis, default 'ms'
%  .YUnit  - unit of y axis, default epo.unit if this field
%                     exists, 'Hz' otherwise
%  .YDir     - 'normal' (low FreqLimuencies at bottom) or 'reverse'
%  .FreqLim     - A vector giving lowest and highest Frequency. If not specified, 
%              [1 size(epo.x,2)] is taken.
%  .PlotRef  - if 1 plot Reference interval (default 0)
%  .RefYPos     - y position of Reference line 
%  .RefWhisker - length of whiskers (vertical lines)
%  .Ref*     - with * in {'LineStyle', 'LineWidth', 'Color'}
%              selects the appearance of the Reference interval.
%  .Colormap - specifies the colormap for depicting amplitude. Give either
%              a string or a x by 3 Color matrix (default 'jet')
%  .CLim   - Define the Color (=amplitude) limits. If empty (default), 
%              limits correspond to the data limits.
%  .CLimPolicy - if 'sym', Color limits are symmetric (so that 0
%              corresponds to the middle of the colormap) (default
%              'normal')
%  .XGrid, ... -  many axis properties can be used in the usual
%                 way
%  .GridOverPatches - if 1 plot grid (default 0)
%  .Title   - title of the plot to be displayed above the axis. 
%             If OPT.title equals 1, the channel label is used.
%  .Title*  - with * in {'Color', 'FontWeight', 'FontSize'}
%             selects the appearance of the title.
%  .YTitle  - if set, the title is displayed within the axis, with its 
%             Y position corresponding to Ytitle (default [])
%  .ZeroLine  - draw an axis along the y-axis at x=0
%  .ZeroLine*  - with * in {'Color','Style'} selects the
%                drawing style of the axes at x=0/y=0

%Output:
% H - Handle to several graphical objects.
%
% See also plotutil_channel1D, plotutil_channel2D


if nargin==0,
  H= opt_catProps(plotutil_channel1D, plotutil_channel2D);
  return
end

if nargin<2,
  clab= 1;
end

if util_getDataDimension(epo)==1
  if ~isempty(varargin)
    opt1D= plotutil_channel1D;
    opt= opt_structToProplist(opt_substruct(opt_proplistToStruct(varargin{:}),opt1D(:,1)));
  else
    opt=varargin;
  end
  H= plotutil_channel1D(epo, clab, opt{:});
else
  if ~isempty(varargin)
    opt2D= plotutil_channel2D;
    opt=opt_structToProplist( opt_substruct(opt_proplistToStruct(varargin{:}),opt2D(:,1)));
  else
    opt=varargin;
  end
  H= plotutil_channel2D(epo, clab, opt{:});
end