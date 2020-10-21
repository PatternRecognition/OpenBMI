function H= plotutil_channel1D(epo, clab, varargin)
%plotutil_channel1D - Plot the Classwise averages of one channel. Takes 1D data,
%i.e. time or frequency data.
%
%Synposis:
% H= plotutil_channel1D(EPO, CLAB, <OPT>)
%
%Input:
% EPO: struct of epoched signals, see makeEpochs
% CLAB: name (or index) of the channel to be plotted.
% OPT: struct or property/value list of optional properties:
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
%
%Output:
% H: Handle to several graphical objects.
%
%Do not call this function directly, rather use the superfunction
%plot_channel.
%
%See also grid_plot.

% 02-2005 Benjamin Blankertz
% 10-2015 Daniel Miklody


props = {'AxisType',                        'box',                '!CHAR';
         'AxisTitle',                       '',                     'CHAR';
         'AxisTitleHorizontalAlignment',    'center',               'CHAR';
         'AxisTitleVerticalAlignment',      'top',                  'CHAR';
         'AxisTitleColor',                  'k',                    'CHAR';
         'AxisTitleFontSize',               get(gca,'FontSize'),    'DOUBLE';
         'AxisTitleFontWeight',             'normal',               'CHAR';
         'Box',                             'on',                   'CHAR';
         'Butterfly'                        0,                      'BOOL';
         'ChannelLineStyleOrder',           {'-','--','-.',':'},    'CELL{CHAR}';
         'ColorOrder',                      get(gca,'ColorOrder'),  'DOUBLE[- 3]';
         'GridOverPatches',                 1,                      'BOOL';
         'Legend',                          1,                      'BOOL';
         'LegendPos',                       'best',                 'CHAR';
         'LineWidth',                       2,                      'DOUBLE';
         'LineStyle',                       '-',                    'CHAR';
         'LineStyleOrder',                  {},                     'CELL{CHAR}'
         'LineWidthOrder',                  [],                     'DOUBLE';
         'LineSpecOrder',                   {},                     'CELL';
         'MultichannelTitleOpts',           {},                     'STRUCT';
         'OversizePlot',                    1,                      'DOUBLE'
         'PlotStat',                        'none',                 'CHAR';
         'RefCol',                          0.75,                   'DOUBLE';
         'RefVSize',                        0.05,                   'DOUBLE';
         'Reset',                           1,                      'BOOL';
         'SmallSetup',                      0,                      'BOOL';
         'StdLineSpec',                     '--',                   'CHAR';
         'ShadeDifference',                 0                       'BOOL';
         'ShadeDifferenceColor',            [1 0.8 1]               'DOUBLE[3]|CHAR';
         'Title',                           1,                      'BOOL';
         'TitleColor',                      'k',                    'CHAR';
         'TitleFontSize',                   get(gca,'FontSize'),    'DOUBLE';
         'TitleFontWeight',                 'normal',               'CHAR';
         'XZeroLine',                       1,                      'DOUBLE';
         'YZeroLine',                       1,                      'DOUBLE';
         'YDir',                            'normal',               'CHAR';
         'XGrid',                           'on',                   'CHAR';
         'YGrid',                           'on',                   'CHAR';
         'XUnit',                           'ms',                   'CHAR';
         'YUnit',                           'a.u.',                 'CHAR';
         'UnitDispPolicy',                  'label',                'CHAR';
         'XUnitDispPolicy',                 'label',                'CHAR';
         'YUnitDispPolicy',                 'label',                'CHAR';
         'YLim',                            [],                     'DOUBLE[2]';
         'YLimPolicy',                      'tight',                'CHAR';
         'ZeroLineColor',                   0.5*[1 1 1],            'DOUBLE[3]';
         'ZeroLineStyle',                   '-',                    'CHAR';
         'ZeroLineTickLength',              3,                      'DOUBLE';
         };

% If the new plotutil_gridOverPatches proved to work well, we can remove
% the following stuff here (BB)
props_gridOverPatches = plotutil_gridOverPatches;

if nargin==0,
  H= opt_catProps(props, props_gridOverPatches); return
end

opt= opt_proplistToStruct(varargin{:});
[opt, isdefault]= opt_setDefaults(opt, props);
opt_checkProplist(opt, props, props_gridOverPatches);

[opt, isdefault]= ...
    opt_overrideIfDefault(opt, isdefault, ...
    'XUnitDispPolicy',opt.UnitDispPolicy,...
    'YUnitDispPolicy',opt.UnitDispPolicy);

if max(sum(epo.y,2))>1,
    if strcmpi(opt.PlotStat,'std'),
        epo= proc_average(epo, 'std',1);
    elseif strcmpi(opt.PlotStat,'sem')
        epo= proc_average(epo, 'Stats',1);
    elseif strcmpi(opt.PlotStat,'perc')
        epo= proc_percentiles(epo, [25 75]);
    else
        epo= proc_average(epo);        
    end    
else
    % epo contains already averages (or single trials)
    % sort Classes
    [tmp,si]= sort([1:size(epo.y,1)]*epo.y);
    epo.y= epo.y(:,si);  % should be an identity matrix now
    epo.x= epo.x(:,:,si);
    if strcmpi(opt.PlotStat,'std')
        if isfield(epo, 'std');
            epo.std= epo.std(:,:,si);
        else
            error('When using opt.PlotStat==std the standard deviation needs to be calculated using proc_average or the unaveraged data has to be given.')
        end
    end
    if strcmpi(opt.PlotStat,'sem')
        if isfield(epo, 'se');
            epo.se= epo.se(:,:,si);
        else
            error('When using opt.PlotStat==sem the se needs to be calculated using proc_average or the unaveraged data has to be given.')
        end
    end
    if strcmpi(opt.PlotStat,'perc')
        if isfield(epo, 'percentiles');
            epo.percentiles.x= epo.percentiles.x(:,:,si,:);
        else
            error('When using opt.PlotStat==proc the percentiles need to be calculated using proc_percentiles.')
        end
    end
end

chan= util_chanind(epo, clab);
nChans= length(chan);
nClasses= size(epo.y, 1);
if nChans==0,
  error('channel ''%s'' not found', clab); 
elseif nChans>1 && opt.Butterfly==0
  if ~isempty(opt.LineStyleOrder),
    error('do not use opt.LineStyleOrder when plotting multi channel');
  end
  opt_plot= {'Reset',1, 'XZeroLine',0, 'YZeroLine',0, 'Title',0, ...
            'GridOverPatches',0};
  tit= cell(1, nChans);
  for ic= 1:nChans,
    if ic==nChans,
      opt_plot([4 6])= {1};
    end
    ils= mod(ic-1, length(opt.ChannelLineStyleOrder))+1;
    if strcmpi(opt.ChannelLineStyleOrder{ils}, 'thick'),
      H{ic}= plotutil_channel1D(epo, chan(ic), opt_rmIfDefault(opt, isdefault), ...
                         opt_plot{:});
    elseif strcmpi(opt.ChannelLineStyleOrder{ils}, 'thin'),
      H{ic}= plotutil_channel1D(epo, chan(ic), opt_rmIfDefault(opt, isdefault), ...
                         opt_plot{:}, 'LineWidth',1);
    else
      H{ic}= plotutil_channel1D(epo, chan(ic), opt_rmIfDefault(opt, isdefault), ...
                         opt_plot{:}, ...
                         'LineStyle',opt.ChannelLineStyleOrder{ils});
    end
    hold on;
    tit{ic}= sprintf('%s (%s)', epo.clab{chan(ic)}, ...
                     opt.ChannelLineStyleOrder{ils});
    opt_plot{2}= 0;
  end
  hold off;
  if opt.Legend,
    H{1}.leg= legend(H{1}.plot, epo.className, opt.LegendPos);
  else
    H{1}.leg= NaN;
  end
  H{1}.title= axis_title(tit, opt.MultichannelTitleOpts{:});
  ud= struct('type','ERP', 'chan',{epo.clab(chan)}, 'hleg',H{1}.leg);
  set(gca, 'userData', ud);
  return;
elseif nChans>1 && opt.Butterfly
  if numel(epo.className)>1
    epo = proc_selectClasses(epo,1); 
    warning('Multiple classes found, plotting only first class..')
  end
end

% Post-process opt properties
if ~iscell(opt.StdLineSpec),
  opt.StdLineSpec= {opt.StdLineSpec};
end
if isequal(opt.Title, 1),
  if opt.Butterfly
     opt.Title= '';
   else
     opt.Title= epo.clab(chan);
   end
end
if opt.SmallSetup,
  if ~isfield(opt, 'XTickLabel') && ~isfield(opt, 'XTickLabelMode'),
    if isfield(opt, 'XTick') || ...
          (isfield(opt, 'XTickMode') && strcmp(opt.xTickMode,'auto')),
      opt.xTickLabelMode= 'auto';
    else
      opt.xTickLabel= [];
    end
  end
  if ~isfield(opt, 'YTickLabel') && ~isfield(opt, 'YTickLabelMode'),
    if isfield(opt, 'YTick') || ...
          (isfield(opt, 'YTickMode') && strcmp(opt.yTickMode,'auto')),
      opt.yTickLabelMode= 'auto';
    else
      opt.yTickLabel= [];
    end
  end
  if isdefault.LineWidth,
    opt.LineWidth= 0.5;
  end
end
if isdefault.XUnit && isfield(epo, 'xUnit'),
  opt.XUnit= epo.xUnit;
end
if isdefault.YUnit && isfield(epo, 'yUnit'),
  opt.YUnit= epo.yUnit;
elseif isdefault.YUnit && isfield(epo, 'cnt_info') && ...
        isfield(epo.cnt_info, 'yUnit');
  opt.YUnit= epo.cnt_info.yUnit;
end
if strcmpi(opt.YUnitDispPolicy, 'lasttick'),
  opt.YUnit= strrep(opt.YUnit, '\mu','u');
end
if strcmpi(opt.AxisType, 'cross'),  % other default values for 'cross'
  [opt,isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'XGrid', 'off', ...
                            'YGrid', 'off', ...
                            'GridOverPatches', 0, ...
                            'xColor', [1 1 1]*0.999, ...
                            'yColor', [1 1 1]*0.999);
  % *0.999 is used, since Matlab prints XColor [1 1 1] as black.
end
if isequal(opt.ColorOrder,'rainbow'),
  ColorOrder_hsv= [(0.5:nClasses)'/nClasses ones(nClasses,1)*[1 0.85]];
  opt.ColorOrder= hsv2rgb(ColorOrder_hsv);
elseif opt.Butterfly && isdefault.ColorOrder
  cm = get(gcf,'colormap');
  cm = repmat(cm,[ceil(numel(epo.clab)/size(cm,1)) 1]); % in case we have more channels than colors
  opt.ColorOrder= cm(1:numel(epo.clab),:);
else
  opt.ColorOrder= opt.ColorOrder(1:min([nClasses size(opt.ColorOrder,1)]),:);
end

[axesStyle, lineStyle]= ...
            opt_extractPlotStyles(rmfield(opt,{'YLim','LineStyleOrder'}));
[opt, isdefault]= opt_setDefaults(opt, {'LineStyleOrder', {};
                  'LineWidthOrder', [];
                  'LineSpecOrder', {}} );
ns= length(opt.LineStyleOrder);
nw= length(opt.LineWidthOrder);
if isempty(opt.LineSpecOrder) && max([ns nw])>0,
  opt.LineSpecOrder= cell(1, nClasses);
  for cc= 1:nClasses,
    lsp= {};
    if ~isempty(opt.LineStyleOrder),
      ii= mod(cc-1, ns)+1;
      lsp= {lsp{:}, 'LineStyle',opt.LineStyleOrder{ii}};
    end
    if ~isempty(opt.LineWidthOrder),
      ii= mod(cc-1, nw)+1;
      lsp= {lsp{:}, 'LineWidth',opt.LineWidthOrder(ii)};
    end
    opt.LineSpecOrder{cc}= lsp;
  end
end
if length(opt.RefCol)==1,
  opt.RefCol= opt.RefCol*[1 1 1];
end

% Set missing optional fields of epo to default values
if ~isfield(epo, 't'),
  epo.t= 1:size(epo.x,1);
end
if ~isfield(epo, 'className'),
  epo.className= cellstr([repmat('Class ',nClasses,1) num2str((1:nClasses)')]);
end

H.ax= gca;
if opt.Reset,
  cla('reset');
end
set(H.ax, axesStyle{:});
hold on;      %% otherwise axis properties like ColorOrder are lost
if opt.ShadeDifference,
  xx= squeeze(epo.x(:,chan,:));
  H.shade= patch([epo.t(:); flipud(epo.t(:))], [xx(:,1); flipud(xx(:,2))], ...
                 opt.ShadeDifferenceColor, 'EdgeColor','none');
end
H.plot= plot(epo.t, squeeze(epo.x(:,chan,:)));
if length(lineStyle)>0,
  set(H.plot, lineStyle{:});
end
for ii= 1:length(opt.LineSpecOrder),
  set(H.plot(ii), opt.LineSpecOrder{ii}{:});
end

if ~strcmpi(opt.PlotStat,'none')
    %prepare data vectors for plotting
    epo.stats=[];
    if strcmpi(opt.PlotStat,'std'),
        epo.stats=[squeeze(epo.x(:,chan,:)-epo.std(:,chan,:)); ...
            flipud(squeeze(epo.x(:,chan,:)+epo.std(:,chan,:)))];
    elseif strcmpi(opt.PlotStat,'sem')
        epo.stats=[squeeze(epo.x(:,chan,:)-epo.se(:,chan,:)); ...
            flipud(squeeze(epo.x(:,chan,:)+epo.se(:,chan,:)))] ;
    elseif strcmpi(opt.PlotStat,'perc')
        %exclude median, becaus this is the basis of the line
        epo.percentiles.x(:,:,:,epo.percentiles.p==50)=[];
        epo.percentiles.p(epo.percentiles.p==50)=[];
        %resort percentiles to have pairs from far to close
        epo.percentiles.p=sort(epo.percentiles.p);
        neworder=[];
        for ii=1:floor(numel(epo.percentiles.p)/2)
            neworder= [neworder ii numel(epo.percentiles.p)+1-ii];
        end
        epo.percentiles.x=epo.percentiles.x(:,:,:,neworder);
        %prepare data vector
        for ii=1:floor(numel(neworder)/2)
            epo.stats(:,:,ii)=[squeeze(epo.percentiles.x(:,chan,:,2*(ii-1)+1));...
                flipud(squeeze(epo.percentiles.x(:,chan,:,2*(ii-1)+2)))] ;
        end        
    end
    %plot every tube individually: for percentiles, more then one tube per
    %class is possible.
    for ii=1:size(epo.stats,2)
        for jj=1:size(epo.stats,3)
            H.plot_stats(jj)=patch([epo.t fliplr(epo.t)], epo.stats(:,ii,jj), ...
                ones(size(epo.stats(:,ii,jj))),...
            'FaceColor',get(H.plot(ii),'Color') ,'FaceAlpha',0.1,'EdgeColor','none');
        end        
    end
    %get grid and plots back on top
    set(gca,'Layer','top')
    for ii=1:size(H.plot,1)
        uistack(H.plot(ii),'top')
    end
end

if ~isempty(opt.YLim),
  yLim= opt.YLim;
else
  if ismember('|',opt.YLimPolicy,'legacy'),
    ii= find(opt.YLimPolicy=='|');
    if strcmp(opt.YLimPolicy(ii+1:end),'sym'),
      opt_selYLim= {'symmetrize',1};
    else
      error('haeh');
    end
    opt.YLimPolicy= opt.YLimPolicy(1:ii-1);
  else
    opt_selYLim= {};
  end
  yLim= visutil_selectYLim(H.ax, 'policy',opt.YLimPolicy, opt_selYLim{:});
end

set(H.ax, axesStyle{:});
hold off;
set(H.ax, 'yLim', yLim);
xLim= epo.t([1 end]);
set(H.ax, 'xLim', xLim);
oldUnits= get(H.ax, 'units');
set(H.ax, 'units', 'pixel');
pos_pixel= get(H.ax, 'position');
set(H.ax, 'units',oldUnits);
H.hidden_objects= [];
if opt.XZeroLine,
  H.xZero= line([-1e10 1e10], [0 0], ...
                'Color',opt.ZeroLineColor, 'lineStyle',opt.ZeroLineStyle);
  H.hidden_objects= [H.hidden_objects; H.xZero];
  if strcmpi(opt.AxisType, 'cross') && opt.ZeroLineTickLength>0,
    xTick= get(H.ax, 'xTick');
    set(H.ax, 'xTickMode','manual');
    tlen= diff(yLim)/pos_pixel(4)*opt.ZeroLineTickLength;
    hl= line([xTick;xTick], [-1;1]*tlen*ones(1,length(xTick)), ...
             'Color',opt.ZeroLineColor);
    H.hidden_objects= [H.hidden_objects; hl];
  end
end
if opt.YZeroLine,
  if strcmpi(opt.AxisType, 'cross'),
    yLim_reduced= yLim + [1 -1]*diff(yLim)*(1-1/opt.OversizePlot)/2;
    if opt.ZeroLineTickLength>0,
      yTick= get(H.ax, 'yTick');
      yTick= yTick(find(yTick>=yLim_reduced(1) & yTick<=yLim_reduced(2)));
      set(H.ax, 'yTickMode','manual');
      tlen= diff(xLim)/pos_pixel(3)*opt.ZeroLineTickLength;
      hl= line([-1;1]*tlen*ones(1,length(yTick)), [yTick;yTick], ...
               'Color',opt.ZeroLineColor);
      H.hidden_objects= [H.hidden_objects; hl];
    end
  else
    yLim_reduced= [-1e10 1e10];
  end
  set(gca,'YLimMode','manual');
  H.yZero= line([0 0], yLim_reduced, ...
                'Color',opt.ZeroLineColor, 'lineStyle',opt.ZeroLineStyle);
  H.hidden_objects= [H.hidden_objects; H.yZero];
end

if isfield(epo, 'refIval'),
  yPatch= [-1 1] * opt.RefVSize * diff(yLim);
  H.refPatch= patch(epo.refIval([1 2 2 1]), yPatch([1 1 2 2]), opt.RefCol);
  set(H.refPatch, 'edgeColor','none');
  obj_moveBack(H.refPatch);
end
if opt.GridOverPatches,
  opt_gridOverPatches = opt_substruct(opt, props_gridOverPatches(:,1));
  plotutil_gridOverPatches(opt_gridOverPatches);
end

switch(lower(opt.XUnitDispPolicy)),
 case 'label',
  H.XLabel= xlabel(['[' opt.XUnit ']']);
%  case 'lasttick',
%   setLastXTickLabel(['[' opt.XUnit ']']);
 case 'none',
  % Not a lot to do here ...
 otherwise,
  error('XUnitDispPolicy unknown');
end
switch(lower(opt.YUnitDispPolicy)),
 case 'label',
  H.YLabel= ylabel(['[' opt.YUnit ']']);
%  case 'lasttick',
%   setLastYTickLabel(['[' opt.YUnit ']']);
 case 'none',
  % Not a lot to do here ...
 otherwise,
  error('YUnitDispPolicy unknown');
end

if opt.Legend && opt.Butterfly==0
  H.leg= legend(H.plot, epo.className, 'Location',opt.LegendPos);
else
  H.leg= NaN;
end

if ~isequal(opt.Title, 0),
  H.title= title(opt.Title);
  set(H.title, 'Color',opt.TitleColor, ...
               'fontWeight',opt.TitleFontWeight, ...
               'FontSize',opt.TitleFontSize);
end

if ~isempty(opt.AxisTitle),
%%% This was necessary for older Matlab versions
%  if strcmp(opt.AxisType, 'cross'),
%    shiftAwayFromBorder= 0;
%  else
%    shiftAwayFromBorder= 0.05;
%  end    
%  switch(opt.AxisTitleHorizontalAlignment),
%   case 'left',
%    xt= xLim(1) + diff(xLim)*shiftAwayFromBorder;
%   case 'center',
%    xt= mean(xLim);
%   case 'right',
%    xt= xLim(2) - diff(xLim)*shiftAwayFromBorder;
%  end
%  yl_axis= yLim + [1 -1]*diff(yLim)*(1-1/opt.OversizePlot)/2;
%  if ismember(opt.AxisTitleVerticalAlignment, {'bottom','baseline'}),
%    yl_axis= yl_axis([2 1]);
%  end
%  yt= yl_axis(2-strcmpi(opt.YDir, 'reverse'));
%  H.ax_title= text(xt, yt, opt.AxisTitle);
%  set(H.ax_title, 'verticalAlignment',opt.AxisTitleVerticalAlignment, ...
%                  'horizontalAlignment',opt.AxisTitleHorizontalAlignment, ...
%                  'Color',opt.AxisTitleColor, ...
%                  'fontWeight',opt.AxisTitleFontWeight, ...
%                  'FontSize',opt.AxisTitleFontSize);
  if strcmp(opt.AxisType, 'cross'),
    shiftAwayFromBorder= 0;
  else
    shiftAwayFromBorder= 0.05;
  end    
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

if ~isempty(H.hidden_objects),
  obj_moveBack(H.hidden_objects);
% If we hide handles, those objects may pop to the front again,
% e.g., when another object is moved to the background with moveObjetBack
%  set(H.hidden_objects, 'handleVisibility','off');
end
ud= struct('type','ERP', 'chan',epo.clab{chan(1)}, 'hleg',H.leg);
set(H.ax, 'userData', ud);

if nargout==0,
  clear H,
end