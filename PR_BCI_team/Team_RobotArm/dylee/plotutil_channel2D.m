function H= plotutil_channel2D(epo, clab, varargin)
%plotutil_channel2D - Plot the Classwise averages of one channel. Plots 2D data,
%i.e. time x frequency.
%
%Usage:
% H= plotutil_channel2D(EPO, CLAB, <OPT>)
%
%Input:
% EPO  - Struct of epoched signals, see makeEpochs
% CLAB - Name (or index) of the channel to be plotted.
% OPT  - struct or property/value list of optional properties:
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
%
%Output:
% H - Handle to several graphical objects.
%
%Do not call this function directly, rather use the superfunction
%plot_channel. This function is an adapted version of plotutil_channel2D.
%
%See also plotutil_channel1D, grid_plot.

% Author(s): Matthias Treder Aug 2010

props = {'AxisType',                  'box',                  'CHAR';
         'AxisTitle',                 '',                     'CHAR';
         'AxisTitleHorizontalAlignment',    'center',               'CHAR';
         'AxisTitleVerticalAlignment',      'top',                  'CHAR';
         'AxisTitleColor',                  'k',                    'CHAR';
         'AxisTitleFontSize',               get(gca,'FontSize'),    'DOUBLE';
         'AxisTitleFontWeight',             'normal',               'CHAR';
         'Box',                       'on',                   'CHAR';
         'ChannelLineStyleOrder',     {'-','--','-.',':'},    'CELL{CHAR}'
         'Colormap',                  'jet',                  'CHAR|DOUBLE[- 3]'
         'CLim',                      [],                     'DOUBLE[2]';
         'CLimPolicy',                'normal',               'CHAR(sym normal)';
         'FreqLim',                   [],                     'DOUBLE[2]';
         'GridOverPatches',           1,                      'BOOL';
         'LineWidth',                 2,                      'DOUBLE';
         'MultichannelTitleOpts',     {},                     'STRUCT';
         'OversizePlot',              1,                      'DOUBLE'
         'PlotRef',                   0,                      'BOOL';
         'RefCol',                    0.75,                   'DOUBLE';
         'RefLineStyle',              '-',                    'CHAR';
         'RefLineWidth',              2,                      'DOUBLE';
         'RefYPos',                   [],                     'DOUBLE';
         'RefWhisker',                [],                     'DOUBLE';
         'SmallSetup',                0,                      'BOOL';
         'Title',                     1,                      'BOOL';
         'TitleColor',                'k',                    'CHAR';
         'TitleFontSize',             get(gca,'FontSize'),    'DOUBLE';
         'TitleFontWeight',           'normal',               'CHAR';
         'XGrid',                     'on',                   'CHAR';
         'XUnit',                     '[ms]',                 'CHAR';
         'YUnit',                     '[Frequency]',          'CHAR';
         'YTitle',                    [],                     'DOUBLE';
         'YDir',                      'normal',               'CHAR';
         'YGrid',                     'on',                   'CHAR';
         'ZeroLine',                  1,                      'DOUBLE';
         'ZeroLineColor',             0.5*[1 1 1],            'DOUBLE[3]';
         'ZeroLineStyle',             '-',                    'CHAR';
          };

if nargin==0,
  H= props; return
end

opt= opt_proplistToStruct(varargin{:});
[opt, isdefault]= opt_setDefaults(opt, props);
opt_checkProplist(opt, props);

if isdefault.FreqLim,
  opt.FreqLim= [1 size(epo.x,2)];
end

if max(sum(epo.y,2))>1,
  epo= proc_average(epo);
else
  % epo contains already averages (or single trials)
  % sort Classes
  [tmp,si]= sort([1:size(epo.y,1)]*epo.y);
  epo.y= epo.y(:,si);  % should be an identity matrix now
  epo.x= epo.x(:,:,:,si);
end

chan= util_chanind(epo, clab);
nChans= length(chan);
nClasses= size(epo.y, 1);

if nChans==0,
  error('channel not found'); 
elseif nChans>1,
  opt_plot= {'ZeroLine',0, 'Title',0, ...
            'GridOverPatches',0};
  tit= cell(1, nChans);
  for ic= 1:nChans,
    if ic==nChans,
      opt_plot([4 6])= {1};
    end
    ils= mod(ic-1, length(opt.ChannelLineStyleOrder))+1;
    H{ic}= plotutil_channel2D(epo, chan(ic), opt_rmIfDefault(opt, isdefault), ...
                       opt_plot{:}, ...
                       'lineStyle',opt.ChannelLineStyleOrder{ils});
    hold on;
    tit{ic}= sprintf('%s (%s)', epo.clab{chan(ic)}, ...
                     opt.ChannelLineStyleOrder{ils});
    opt_plot{2}= 0;
  end
  hold off;
  H{1}.leg= NaN;
  H{1}.title= axis_title(tit, opt.MultichannelTitleOpts{:});
  ud= struct('type','ERP', 'chan',{epo.clab(chan)}, 'hleg',H{1}.leg);
  set(gca, 'userData', ud);
  return;
end

%% Post-process opt properties
if isequal(opt.Title, 1),
  opt.Title= epo.clab(chan);
end

if isdefault.XUnit && isfield(epo, 'xUnit'),
  opt.XUnit= ['[' epo.xUnit ']'];
end
if isdefault.YUnit && isfield(epo, 'yUnit'),
  opt.YUnit= ['[' epo.yUnit ']'];
end

if isdefault.RefYPos,
  opt.RefYPos = opt.FreqLim(1) + .9 * diff(opt.FreqLim);
end

if isdefault.RefWhisker,
  opt.RefWhisker = .05 * diff(opt.FreqLim);
end

if length(opt.RefCol)==1,
  opt.RefCol= opt.RefCol*[1 1 1];
end

%% Set missing optional fields of epo to default values
if ~isfield(epo, 't'),
  epo.t= 1:size(epo.x,1);
end

%% Plot data, zero line, ref ival, grid
H.ax= gca; cla

if ~isempty(opt.CLim)
  if strcmp(opt.CLimPolicy,'sym')  % make Color limits symmetric
    cm = abs(max(opt.CLim));
    opt.CLim = [-cm cm];
  end
  H.plot= imagesc([epo.t(1) epo.t(end)],opt.FreqLim, squeeze(epo.x(:,:,chan,:))', ...
    opt.CLim);
else
  H.plot= imagesc([epo.t(1) epo.t(end)],opt.FreqLim, squeeze(epo.x(:,:,chan,:))');

end
hold on;      

if opt.ZeroLine,
  line([0 0], get(gca,'YLim'), ...
                'Color',opt.ZeroLineColor, 'lineStyle',opt.ZeroLineStyle);
end

% Plot ref ival
if opt.PlotRef && isfield(epo, 'refIval'),
  xx = epo.refIval;
  yy = opt.RefYPos * [1 1];
  lopt = {'LineStyle',opt.RefLineStyle, 'LineWidth',opt.RefLineWidth, ...
    'Color', opt.RefCol};
  line(xx,yy,lopt{:});
  line([xx(1) xx(1)],[yy(1)-opt.RefWhisker yy(1)+opt.RefWhisker], ...
    lopt{:});
  line([xx(end) xx(end)],[yy(1)-opt.RefWhisker yy(1)+opt.RefWhisker], ...
    lopt{:});
end

if opt.GridOverPatches,
  opt_grid= plotutil_gridOverPatches;
  opt_grid= opt_structToProplist(opt_substruct(opt,opt_grid(:,1)));
  plotutil_gridOverPatches(opt_grid{:});
end

%% More layout settings
colormap(opt.Colormap);
set(gca,'YDir',opt.YDir)
H.leg = NaN;

%% title and labels
if ~isequal(opt.Title, 0),
  if isempty(opt.YTitle)
    H.title= title(opt.Title);
    set(H.title, 'Color',opt.TitleColor, ...
                 'fontWeight',opt.TitleFontWeight, ...
                 'FontSize',opt.TitleFontSize);
  else
    xx = sum(get(gca,'XLim'))/2; % middle
    H.title = text(xx,opt.Ytitle, opt.Title, ...
                 'Color',opt.TitleColor, ...
                 'fontWeight',opt.TitleFontWeight, ...
                 'FontSize',opt.TitleFontSize, ...
                 'HorizontalAlignment','center');
    
  end
end

if ~isempty(opt.AxisTitle),
  shiftAwayFromBorder= 0.05;
  switch(opt.AxisTitleHorizontalAlignment),
   case 'left',
    xt= shiftAwayFromBorder;
   case 'center',
    xt= 0.5;
   case 'right',
    xt= 1 - shiftAwayFromBorder;
  end
  yl_axis= [0.01 0.99] + [1 -1]*(1-1/opt.OversizePlot)/2;
  if ismember(opt.AxisTitleVerticalAlignment, {'bottom','baseline'},'legacy'),
    yl_axis= yl_axis([2 1]);
  end
  yt= yl_axis(2-strcmpi(opt.YDir, 'reverse'));
  H.ax_title= title(opt.AxisTitle);
  set(H.ax_title, 'Units','normalized', ...
                  'Position', [xt yt 0]);
  set(H.ax_title, 'verticalAlignment',opt.AxisTitleVerticalAlignment, ...
                  'horizontalAlignment',opt.AxisTitleHorizontalAlignment, ...
                  'Color',opt.AxisTitleColor, ...
                  'fontWeight',opt.AxisTitleFontWeight, ...
                  'FontSize',opt.AxisTitleFontSize);
end


H.xlabel = xlabel(opt.XUnit);
H.ylabel = ylabel(opt.YUnit);

% Plot frequencies if specified
if isfield(epo,'f')
  lab= epo.f(get(H.ax,'Ytick'));
  set(H.ax,'YTickLabel',lab); 
end

% if ~isempty(H.hidden_objects),
%   obj_moveBack(H.hidden_objects);
% % If we hide handles, those objects may pop to the front again,
% % e.g., when another object is moved to the background with moveObjetBack
% %  set(H.hidden_objects, 'handleVisibility','off');
% end
ud= struct('type','ERP', 'chan',epo.clab{chan}, 'hleg',H.leg);
set(H.ax, 'userData', ud);

if nargout==0,
  clear H,
end