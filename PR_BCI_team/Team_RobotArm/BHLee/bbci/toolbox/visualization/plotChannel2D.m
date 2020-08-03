function H= plotChannel2D(epo, clab, varargin)
%plotChannel2D - Plot the classwise averages of one channel. Takes 2D data,
%i.e. time x amplitude or frequency x amplitude.
%
%Usage:
% H= plotChannel2D(EPO, CLAB, <OPT>)
%
%Input:
% EPO  - Struct of epoched signals, see makeEpochs
% CLAB - Name (or index) of the channel to be plotted.
% OPT  - struct or property/value list of optional properties:
%  .plotStd - if true, the standard deviation 
%  .legend - show class legend (1, default), or not (0).
%  .legendPos - position of the legend, see help of function 'legend'.
%  .xUnit  - unit of x axis, default 'ms'
%  .yUnit  - unit of y axis, default epo.unit if this field
%                     exists, '\muV' otherwise
%  .unitDispPolicy - the units can either ('label', default) be displayed as 
%             xlabel resp ylabel, or ('lastTick') instead of the label of 
%             the last (x- resp. y-) tick
%  .yDir   -  'normal' (negative down) or 'reverse' (negative up)
%  .refCol -  color of patch indicating the baseline interval
%  .refVSize - (value, default 0.05) controls the height of the patch
%             marking the reference interval.
%  .colorOrder - specifies the colors for drawing the curves of the
%             different classes. If not given the colorOrder
%             of the current axis is taken. As special gimmick
%             you can use 'rainbow' as colorOrder.
%  .lineWidthOrder - (numerical vector) analog to colorOrder.
%  .lineStyleOrder - (cell array of strings) analog to colorOrder.
%  .lineSpecOrder - (cell array of cell array) analog to colorOrder,
%             giving full control on the appearance of the curves.
%             (If lineSpecOrder is defined, lineWidthOrder and lineStyleOrder
%             are ignored, but colorOrder is not.)
%  .xGrid, ... -  many axis properties can be used in the usual
%                 way
%  .yLim - Define the y limits. If empty (default) an automatic selection
%             according to 'yLimPolicy' is performed.
%  .yLimPolicy - policy of how to select the YLim (only if 'yLim' is
%             empty resp unspecified): 'auto' uses the
%             usual mechanism of Matlab; 'tightest' selects YLim as the
%             exact data range; 'tight' (default) takes the data range,
%             adds a little border and selects 'nice' limit values.
%  .title   - Title of the plot to be displayed above the axis. 
%             If OPT.title equals 1, the channel label is used.
%  .title*  - with * in {'Color', 'FontWeight', 'FontSize'}
%             selects the appearance of the title.
%  .xZeroLine  - draw an axis along the x-axis at y=0
%  .yZeroLine  - draw an axis along the y-axis at x=0
%  .zeroLine*  - with * in {'Color','Style'} selects the
%                drawing style of the axes at x=0/y=0
%  .axisTitle  - (string) title to be displayed *within* the axis.
%  .axisTitle* - with * in {'Color', 'HorizontalAlignment',
%                 'VerticalAlignment', 'FontWeight', 'FontSize'}
%                 selects the appearance of the subplot titles.
%
%Output:
% H - Handle to several graphical objects.
%
%Do not call this function directly, rather use the superfunction
%plotChannel.
%
%See also grid_plot.


% Author(s): Benjamin Blankertz, Sep 2000 / Feb 2005

opt= propertylist2struct(varargin{:});
[opt_orig, isdefault]= ...
    set_defaults(opt, ...
                 'axisType', 'box', ...
                 'legend', 1, ...
                 'legendPos', 0, ...
                 'yDir', 'normal', ...
                 'xGrid', 'on', ...
                 'yGrid', 'on', ...
                 'box', 'on', ...
                 'xUnit', '[ms]', ...
                 'yUnit', '[\muV]', ...
                 'unitDispPolicy', 'label', ...
                 'yLimPolicy', 'tight', ...
                 'refCol', 0.75, ...
                 'refVSize', 0.05, ...
                 'xZeroLine', 1, ...
                 'yZeroLine', 1, ...
                 'zeroLineColor', 0.5*[1 1 1], ...
                 'zeroLineStyle', '-', ...
                 'zeroLineTickLength', 3, ...
                 'reset', 1, ...
                 'lineWidth', 2, ...
                 'channelLineStyleOrder', {'-','--','-.',':'}, ...
                 'title', 1, ...
                 'titleColor', 'k', ...
                 'titleFontSize', get(gca,'fontSize'), ...
                 'titleFontWeight', 'normal', ...
                 'smallSetup', 0, ...
                 'axisTitle', '', ...
                 'axisTitleHorizontalAlignment', 'center', ...
                 'axisTitleVerticalAlignment', 'top', ...
                 'axisTitleColor', 'k', ...
                 'axisTitleFontSize', get(gca,'fontSize'), ...
                 'axisTitleFontWeight', 'normal', ...
                 'multichannel_title_opts', {}, ...
                 'colorOrder', get(gca,'colorOrder'), ...
                 'grid_over_patches', 1, ...
                 'plotStd', 0, ...
                 'stdLineSpec', '--', ...
                 'oversizePlot',1);
opt= setfield(opt_orig, 'fig_hidden', 1);
[opt, isdefault]= ...
    opt_overrideIfDefault(opt, isdefault, ...
                          'xUnitDispPolicy',opt.unitDispPolicy,...
                          'yUnitDispPolicy',opt.unitDispPolicy);

if max(sum(epo.y,2))>1,
  epo= proc_average(epo, 'std',opt.plotStd);
else
  % epo contains already averages (or single trials)
  % sort classes
  [tmp,si]= sort([1:size(epo.y,1)]*epo.y);
  epo.y= epo.y(:,si);  % should be an identity matrix now
  epo.x= epo.x(:,:,si);
  if isfield(epo, 'std');
    epo.std= epo.std(:,:,si);
  end
end

chan= chanind(epo, clab);
nChans= length(chan);
nClasses= size(epo.y, 1);
if nChans==0,
  error('channel not found'); 
elseif nChans>1,
  if isfield(opt, 'lineStyleOrder'),
    error('do not use opt.lineStyleOrder when plotting multi channel');
  end
  opt_plot= {'reset',1, 'xZeroLine',0, 'yZeroLine',0, 'title',0, ...
            'grid_over_patches',0};
  tit= cell(1, nChans);
  for ic= 1:nChans,
    if ic==nChans,
      opt_plot([4 6])= {1};
    end
    ils= mod(ic-1, length(opt.channelLineStyleOrder))+1;
    if strcmpi(opt.channelLineStyleOrder{ils}, 'thick'),
      H{ic}= plotChannel2D(epo, chan(ic), opt_rmifdefault(opt, isdefault), ...
                         opt_plot{:});
    elseif strcmpi(opt.channelLineStyleOrder{ils}, 'thin'),
      H{ic}= plotChannel2D(epo, chan(ic), opt_rmifdefault(opt, isdefault), ...
                         opt_plot{:}, 'LineWidth',1);
    else
      H{ic}= plotChannel2D(epo, chan(ic), opt_rmifdefault(opt, isdefault), ...
                         opt_plot{:}, ...
                         'lineStyle',opt.channelLineStyleOrder{ils});
    end
    hold on;
    tit{ic}= sprintf('%s (%s)', epo.clab{chan(ic)}, ...
                     opt.channelLineStyleOrder{ils});
    opt_plot{2}= 0;
  end
  hold off;
  if opt.legend,
    H{1}.leg= legend(H{1}.plot, epo.className, opt.legendPos);
  else
    H{1}.leg= NaN;
  end
  H{1}.title= axis_title(tit, opt.multichannel_title_opts{:});
  ud= struct('type','ERP', 'chan',{epo.clab(chan)}, 'hleg',H{1}.leg);
  set(gca, 'userData', ud);
  return;
end

%% Post-process opt properties
if ~iscell(opt.stdLineSpec),
  opt.stdLineSpec= {opt.stdLineSpec};
end
if isequal(opt.title, 1),
  opt.title= epo.clab(chan);
end
if opt.smallSetup,
  if ~isfield(opt, 'xTickLabel') & ~isfield(opt, 'xTickLabelMode'),
    if isfield(opt, 'xTick') | ...
          (isfield(opt, 'xTickMode') & strcmp(opt.xTickMode,'auto')),
      opt.xTickLabelMode= 'auto';
    else
      opt.xTickLabel= [];
    end
  end
  if ~isfield(opt, 'yTickLabel') & ~isfield(opt, 'yTickLabelMode'),
    if isfield(opt, 'yTick') | ...
          (isfield(opt, 'yTickMode') & strcmp(opt.yTickMode,'auto')),
      opt.yTickLabelMode= 'auto';
    else
      opt.yTickLabel= [];
    end
  end
  if isdefault.lineWidth,
    opt.lineWidth= 0.5;
  end
end
if isdefault.xUnit & isfield(epo, 'xUnit'),
  opt.xUnit= ['[' epo.xUnit ']'];
end
if isdefault.yUnit & isfield(epo, 'yUnit'),
  opt.yUnit= ['[' epo.yUnit ']'];
end
if strcmpi(opt.yUnitDispPolicy, 'lasttick'),
  opt.yUnit= strrep(opt.yUnit, '\mu','u');
end
if strcmpi(opt.axisType, 'cross'),  %% other default values for 'cross'
  [opt,isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'xGrid', 'off', ...
                            'yGrid', 'off', ...
                            'grid_over_patches', 0, ...
                            'xColor', [1 1 1]*0.999, ...
                            'yColor', [1 1 1]*0.999);
  %% *0.999 is used, since Matlab prints XColor white as black.
end
if isequal(opt.colorOrder,'rainbow'),
  colorOrder_hsv= [(0.5:nClasses)'/nClasses ones(nClasses,1)*[1 0.85]];
  opt.colorOrder= hsv2rgb(colorOrder_hsv);
else
  opt.colorOrder= opt.colorOrder(1:min([nClasses size(opt.colorOrder,1)]),:);
end
[axesStyle, lineStyle]= opt_extractPlotStyles(opt);
[opt, isdefault]= set_defaults(opt, ...
                  'lineStyleOrder', {}, ...
                  'lineWidthOrder', [], ...
                  'lineSpecOrder', {});
ns= length(opt.lineStyleOrder);
nw= length(opt.lineWidthOrder);
if isempty(opt.lineSpecOrder) & max([ns nw])>0,
  opt.lineSpecOrder= cell(1, nClasses);
  for cc= 1:nClasses,
    lsp= {};
    if ~isempty(opt.lineStyleOrder),
      ii= mod(cc-1, ns)+1;
      lsp= {lsp{:}, 'LineStyle',opt.lineStyleOrder{ii}};
    end
    if ~isempty(opt.lineWidthOrder),
      ii= mod(cc-1, nw)+1;
      lsp= {lsp{:}, 'LineWidth',opt.lineWidthOrder(ii)};
    end
    opt.lineSpecOrder{cc}= lsp;
  end
end
if length(opt.refCol)==1,
  opt.refCol= opt.refCol*[1 1 1];
end

%% Set missing optional fields of epo to default values
if ~isfield(epo, 't'),
  epo.t= 1:size(epo.x,1);
end
if ~isfield(epo, 'className'),
  epo.className= cellstr([repmat('class ',nClasses,1) num2str((1:nClasses)')]);
end

H.ax= gca;
if opt.reset,
  cla('reset');
end
set(H.ax, axesStyle{:});
hold on;      %% otherwise axis properties like colorOrder are lost
H.plot= plot(epo.t, squeeze(epo.x(:,chan,:)));
if length(lineStyle)>0,
  set(H.plot, lineStyle{:});
end
for ii= 1:length(opt.lineSpecOrder),
  set(H.plot(ii), opt.lineSpecOrder{ii}{:});
end
if isfield(opt, 'yLim'),
  yLim= opt.yLim;
else
  if ismember('|',opt.yLimPolicy),
    ii= find(opt.yLimPolicy=='|');
    if strcmp(opt.yLimPolicy(ii+1:end),'sym'),
      opt_selYLim= {'symmetrize',1};
    else
      error('haeh');
    end
    opt.yLimPolicy= opt.yLimPolicy(1:ii-1);
  else
    opt_selYLim= {};
  end
  yLim= selectYLim(H.ax, 'policy',opt.yLimPolicy, opt_selYLim{:});
end

if opt.plotStd,
  if iscell(opt.stdLineSpec{1}),
    H.plot_std= plot(epo.t, [squeeze(epo.x(:,chan,:)-epo.std(:,chan,:)) ...
                             squeeze(epo.x(:,chan,:)+epo.std(:,chan,:))]);
    for cc= 1:length(H.plot_std),
      set(H.plot_std(cc), opt.stdLineSpec{cc}{:});
    end
  else
    H.plot_std= plot(epo.t, [squeeze(epo.x(:,chan,:)-epo.std(:,chan,:)) ...
                             squeeze(epo.x(:,chan,:)+epo.std(:,chan,:))], ...
                     opt.stdLineSpec{:});
  end
end

set(H.ax, axesStyle{:});
hold off;
xLim= epo.t([1 end]);
set(H.ax, 'xLim', xLim);

oldUnits= get(H.ax, 'units');
set(H.ax, 'units', 'pixel');
pos_pixel= get(H.ax, 'position');
set(H.ax, 'units',oldUnits);
H.hidden_objects= [];
if opt.xZeroLine,
  H.xZero= line([-1e10 1e10], [0 0], ...
                'color',opt.zeroLineColor, 'lineStyle',opt.zeroLineStyle);
  H.hidden_objects= [H.hidden_objects; H.xZero];
  if strcmpi(opt.axisType, 'cross') & opt.zeroLineTickLength>0,
    xTick= get(H.ax, 'xTick');
    set(H.ax, 'xTickMode','manual');
    tlen= diff(yLim)/pos_pixel(4)*opt.zeroLineTickLength;
    hl= line([xTick;xTick], [-1;1]*tlen*ones(1,length(xTick)), ...
             'color',opt.zeroLineColor);
    H.hidden_objects= [H.hidden_objects; hl];
  end
end
if opt.yZeroLine,
  if strcmpi(opt.axisType, 'cross'),
    yLim_reduced= yLim + [1 -1]*diff(yLim)*(1-1/opt.oversizePlot)/2;
    if opt.zeroLineTickLength>0,
      yTick= get(H.ax, 'yTick');
      yTick= yTick(find(yTick>=yLim_reduced(1) & yTick<=yLim_reduced(2)));
      set(H.ax, 'yTickMode','manual');
      tlen= diff(xLim)/pos_pixel(3)*opt.zeroLineTickLength;
      hl= line([-1;1]*tlen*ones(1,length(yTick)), [yTick;yTick], ...
               'color',opt.zeroLineColor);
      H.hidden_objects= [H.hidden_objects; hl];
    end
  else
    yLim_reduced= [-1e10 1e10];
  end
  H.yZero= line([0 0], yLim_reduced, ...
                'color',opt.zeroLineColor, 'lineStyle',opt.zeroLineStyle);
  H.hidden_objects= [H.hidden_objects; H.yZero];
end

if isfield(epo, 'refIval'),
  yPatch= [-1 1] * opt.refVSize * diff(yLim);
  H.refPatch= patch(epo.refIval([1 2 2 1]), yPatch([1 1 2 2]), opt.refCol);
  set(H.refPatch, 'edgeColor','none');
  moveObjectBack(H.refPatch);
end
if opt.grid_over_patches,
  grid_over_patches(copy_struct(opt, 'xGrid','yGrid'));
end

switch(lower(opt.xUnitDispPolicy)),
 case 'label',
  H.XLabel= xlabel(opt.xUnit);
 case 'lasttick',
  setLastXTickLabel(opt.xUnit);
 case 'none',
  %% Not a lot to do here ...
 otherwise,
  error('xUnitDispPolicy unknown');
end
switch(lower(opt.yUnitDispPolicy)),
 case 'label',
  H.YLabel= ylabel(opt.yUnit);
 case 'lasttick',
  setLastYTickLabel(opt.yUnit);
 case 'none',
  %% Not a lot to do here ...
 otherwise,
  error('yUnitDispPolicy unknown');
end

if opt.legend,
  H.leg= legend(H.plot, epo.className, opt.legendPos);
else
  H.leg= NaN;
end

if ~isequal(opt.title, 0),
  H.title= title(opt.title);
  set(H.title, 'color',opt.titleColor, ...
               'fontWeight',opt.titleFontWeight, ...
               'fontSize',opt.titleFontSize);
end

if ~isempty(opt.axisTitle),
%% This was necessary for older Matlab versions
%  if strcmp(opt.axisType, 'cross'),
%    shiftAwayFromBorder= 0;
%  else
%    shiftAwayFromBorder= 0.05;
%  end    
%  switch(opt.axisTitleHorizontalAlignment),
%   case 'left',
%    xt= xLim(1) + diff(xLim)*shiftAwayFromBorder;
%   case 'center',
%    xt= mean(xLim);
%   case 'right',
%    xt= xLim(2) - diff(xLim)*shiftAwayFromBorder;
%  end
%  yl_axis= yLim + [1 -1]*diff(yLim)*(1-1/opt.oversizePlot)/2;
%  if ismember(opt.axisTitleVerticalAlignment, {'bottom','baseline'}),
%    yl_axis= yl_axis([2 1]);
%  end
%  yt= yl_axis(2-strcmpi(opt.yDir, 'reverse'));
%  H.ax_title= text(xt, yt, opt.axisTitle);
%  set(H.ax_title, 'verticalAlignment',opt.axisTitleVerticalAlignment, ...
%                  'horizontalAlignment',opt.axisTitleHorizontalAlignment, ...
%                  'color',opt.axisTitleColor, ...
%                  'fontWeight',opt.axisTitleFontWeight, ...
%                  'fontSize',opt.axisTitleFontSize);
  if strcmp(opt.axisType, 'cross'),
    shiftAwayFromBorder= 0;
  else
    shiftAwayFromBorder= 0.05;
  end    
  switch(opt.axisTitleHorizontalAlignment),
   case 'left',
    xt= shiftAwayFromBorder;
   case 'center',
    xt= 0.5;
   case 'right',
    xt= 1 - shiftAwayFromBorder;
  end
  yl_axis= [0.01 0.99] + [1 -1]*(1-1/opt.oversizePlot)/2;
  if ismember(opt.axisTitleVerticalAlignment, {'bottom','baseline'}),
    yl_axis= yl_axis([2 1]);
  end
  yt= yl_axis(2-strcmpi(opt.yDir, 'reverse'));
  H.ax_title= title(opt.axisTitle);
  set(H.ax_title, 'Units','normalized', ...
                  'Position', [xt yt 0]);
  set(H.ax_title, 'verticalAlignment',opt.axisTitleVerticalAlignment, ...
                  'horizontalAlignment',opt.axisTitleHorizontalAlignment, ...
                  'color',opt.axisTitleColor, ...
                  'fontWeight',opt.axisTitleFontWeight, ...
                  'fontSize',opt.axisTitleFontSize);
end

if ~isempty(H.hidden_objects),
  moveObjectBack(H.hidden_objects);
% If we hide handles, those objects may pop to the front again,
% e.g., when another object is moved to the background with moveObjetBack
%  set(H.hidden_objects, 'handleVisibility','off');
end
ud= struct('type','ERP', 'chan',epo.clab{chan}, 'hleg',H.leg);
set(H.ax, 'userData', ud);

if nargout==0,
  clear H,
end
