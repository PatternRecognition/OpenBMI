function H= grid_plot(epo, mnt, varargin)
%h= grid_plot(epo, mnt, <opt>)
%
% avoid the title line by setting opt.titleDir= 'none'.
%
% IN   epo     - struct of epoched signals, see makeSegments
%      mnt     - struct for electrode montage, see setElectrodeMontage
%      opt - property/value list or struct of options with fields/properties:
%           .xUnit  - unit of x axis, default 'ms'
%           .yUnit  - unit of y axis, default epo.unit if this field
%                     exists, '\muV' otherwise
%           .yDir   - 'normal' (negative down) or 'reverse' (negative up)
%           .refCol - color of patch indicating the baseline interval
%           .colorOrder -  defines the colors for drawing the curves of the
%                          different classes. if not given the colorOrder
%                          of the current axis is taken. as special gimmick
%                          you can use 'rainbow' as colorOrder.
%           .xGrid, ... -  many axis properties can be used in the usual
%                          way
%           .scaleGroup -  groups of channels, where each group should
%                          get the same y limits, cell array of cells,
%                          default {scalpChannels, {'EMG*'},{'EOGh'},{'EOGv'}};
%           .scalePolicy - says how the y limits are chosen:
%                          'auto': choose automatically,
%                          'sym': automatically but symmetric around 0
%                          [lower upper]: define y limits;
%                          .scalePolicy can also be a cell array, where
%                          each cell corresponds to one .scaleGroup
%           .xZeroLine   - draw an axis along the x-axis at y=0
%           .yZeroLine   - draw an axis along the y-axis at x=0
%           .zeroLine*   - with * in {'Color','Style'} selects the
%                          drawing style of the axes at x=0/y=0
%           .titleDir    - direction of figure title,
%                          'horizontal', 'vertical', 'none'
%           .axisTitle*  - with * in {'Color', 'HorizontalAlignment',
%                          'VerticalAlignment', 'FontWeight', 'FontSize'}
%                          selects the appearance of the subplot titles.
%
% H: handle to several graphical objects
%
% SEE  makeEpochs, mnt_setGrid, scalpChannels, grid_*

% Author(s): Benjamin Blankertz, Feb 2003

opt= propertylist2struct(varargin{:});
[opt,isdefault]= ...
    set_defaults(opt, ...
                 'yDir', 'normal', ...
                 'xUnit', 'ms', ...
                 'yUnit', '\muV', ...
                 'refCol', 0.8, ...
                 'yAxis', 'tight', ...
                 'xGrid', 'on', ...
                 'yGrid', 'on', ...
                 'axisType', 'box', ...
                 'shrinkAxes', [1 1], ...
                 'shiftAxesUp', [], ...
                 'oversizePlot', 1, ...
                 'xZeroLine', 'on', ...
                 'yZeroLine', 'on', ...
                 'zeroLineColor', 0.5*[1 1 1], ...
                 'zeroLineStyle', '-', ...
                 'zeroLineTickLength', 3, ...
                 'scalePolicy', 'auto', ...
                 'legendVerticalAlignment', 'middle', ...
                 'figure_color', [0.8 0.8 0.8], ...
                 'titleDir', 'horizontal', ...
                 'axisTitleHorizontalAlignment', 'center', ...
                 'axisTitleVerticalAlignment', 'top', ...
                 'axisTitleColor', 'k', ...
                 'axisTitleFontSize', get(gca,'fontSize'), ...
                 'axisTitleFontWeight', 'normal', ...
                 'scale_showOrientation', 1);

if strcmpi(opt.axisType, 'cross'),  %% other default values for 'cross'
  [opt, isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'shrinkAxes', [0.9 0.9], ...
                            'xColor', [1 1 1], ...
                            'yColor', [1 1 1], ...
                            'axisTitleVerticalAlignment', 'cap');
%                            'oversizePlot', 1.5, ...
end

if opt.oversizePlot>1,
  [opt, isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'visible', 'off');
end
if isfield(opt, 'xTick'),
  [opt, isdefault]= ...
      opt_overrideIfDefault(opt, isdefault, ...
                            'shrinkAxes', 0.8);
  if ~isfield(opt, 'xTickLabel'),
    opt.xTickLabelMode= 'auto';
  end
end
if isfield(opt, 'yTick') & ~isfield(opt, 'yTickLabel'),
  opt.yTickLabelMode= 'auto';
end

if ~isfield(opt, 'xTickLabel') & ...
      (~isfield(opt, 'xTickMode') | ~strcmp(opt.xTickMode,'auto')), 
  opt.xTickLabel= []; 
end
if ~isfield(opt, 'yTickLabel') & ...
      (~isfield(opt, 'yTickMode') | ~strcmp(opt.yTickMode,'auto')), 
  opt.yTickLabel= []; 
end

if isdefault.shiftAxesUp & ...
      (isfield(opt, 'xTick') & ~isempty(opt.xTick)), ...
      opt.shiftAxesUp= 0.05;
end

if isdefault.xUnit & isfield(epo, 'xUnit'),
  opt.xUnit= epo.xUnit;
end
if isdefault.yUnit & isfield(epo, 'yUnit'),
  opt.yUnit= epo.yUnit;
end
if isfield(opt, 'colorOrder') & isequal(opt.colorOrder,'rainbow'),
  nChans= size(epo.y,1);
  opt.colorOrder= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
end
if ~isfield(opt, 'scaleGroup'),
  scalp_idx= scalpChannels(epo);
  emgeog_idx= chanind(epo, 'EOGh','EOGv','EMG*');
  others_idx= setdiff(1:length(epo.clab), [scalp_idx emgeog_idx]);
  opt.scaleGroup= {epo.clab(scalp_idx), {'EMG*'}, {'EOGh'}, {'EOGv'}, ...
                  epo.clab(others_idx)};
  opt.scalePolicy= {'auto', [-5 50], 'sym', 'auto', 'auto'};
end
if ischar(opt.scalePolicy) | isnumeric(opt.scalePolicy),
  opt.scalePolicy= repmat({opt.scalePolicy}, 1, length(opt.scaleGroup));
end
if length(opt.shrinkAxes)==1,
  opt.shrinkAxes= [1 opt.shrinkAxes];
end

mnt.box_sz(1,:)= mnt.box_sz(1,:) * opt.shrinkAxes(1);
mnt.box_sz(2,:)= mnt.box_sz(2,:) * opt.shrinkAxes(2) * opt.oversizePlot;
if isfield(mnt, 'scale_box_sz'),
  mnt.scale_box_sz(1)= mnt.scale_box_sz(1)*opt.shrinkAxes(1);
  mnt.scale_box_sz(2)= mnt.scale_box_sz(2)*opt.shrinkAxes(1)*opt.oversizePlot;
end

clf;
set(gcf, 'color',opt.figure_color);

flags= {'legend', 'grid'};
[axesStyle, lineStyle]= opt_extractPlotStyles(opt);

dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
if isfield(mnt, 'box'),
  dispChans= intersect(dispChans, find(~isnan(mnt.box(1,1:end-1))));
end
nDisps= length(dispChans);
H.ax= zeros(1, nDisps);
axesTitle= cell(1, nDisps);
for ia= 1:nDisps,
  ic= dispChans(ia);
  H.ax(ia)= axes('position', getAxisGridPos(mnt, ic));
  set(H.ax(ia), axesStyle{:});
  hold on;      %% otherwise axis properties like colorOrder are lost
  [nEvents, hp, dmy, hleg]= showERP(epo, mnt, mnt.clab{ic}, flags{:});
  if length(lineStyle)>0,
    set(hp, lineStyle{:});
  end
  hold off; box on;
  set(H.ax(ia), axesStyle{:});
  if ic==dispChans(1),
    flags= {flags{2:end}}; 
    H.leg= hleg;
    leg_pos= getAxisGridPos(mnt, 0);
    if ~any(isnan(leg_pos)),
      leg_pos_orig= get(hleg, 'position');
      if leg_pos(4)>leg_pos_orig(4),
        switch(lower(opt.legendVerticalAlignment)), 
          case 'top',
           leg_pos(2)= leg_pos(2)-leg_pos_orig(4)+leg_pos(4);
         case 'middle',
           leg_pos(2)= leg_pos(2)+(leg_pos(4)-leg_pos_orig(4))/2;
        end
      end
      leg_pos(3:4)= leg_pos_orig(3:4);  %% use original size
      set(hleg, 'position', leg_pos);
      ud= get(H.leg, 'userData');
      ud= set_defaults(ud, 'type','ERP plus', 'chan','legend');
      set(H.leg, 'visible','off', 'userData',ud);
      set(get(H.leg,'children'), 'visible','on');
    end
  end
%  set(H.ax(ia), 'position', getAxisHeadPos(mnt, ic, axisSize));
%  axis off;
%% mnt.clab{ic} may differ from epo.clab{io}, e.g. the former
%% may be 'C3' while the latter is 'C3 lap'
  io= chanind(epo, mnt.clab{ic});
  axesTitle{ia}= epo.clab{io};
end

%w_cm= warning('query', 'bci:missing_channels');
%warning('off', 'bci:missing_channels');
%all_idx= 1:length(mnt.clab);
yLim= zeros(length(opt.scaleGroup), 2);
for ig= 1:length(opt.scaleGroup),
  ax_idx= chanind(mnt.clab(dispChans), opt.scaleGroup{ig});
  if isempty(ax_idx), continue; end
%  ch_idx= find(ismember(all_idx, ax_idx));
  ch_idx= dispChans(ax_idx);
  if isnumeric(opt.scalePolicy{ig}),
    yLim(ig,:)= opt.scalePolicy{ig};
  else
    yLim(ig,:)= unifyYLim(H.ax(ax_idx), opt.yAxis);
  end
  if isequal(opt.scalePolicy{ig},'sym'),
    yl= max(abs(yLim(ig,:)));
    yLim(ig,:)= [-yl yl];
  end
  if ig==1,
    scale_with_group1= setdiff(1:nDisps, chanind(mnt.clab(dispChans), ...
                                                 [opt.scaleGroup{2:end}]));
    set(H.ax(scale_with_group1), 'yLim',yLim(ig,:));
    ch2group= ones(1,nDisps);
  else
    set(H.ax(ax_idx), 'yLim',yLim(ig,:));
    ch2group(ax_idx)= ig;
    for ia= ax_idx,
      if max(abs(yLim(ig,:)))>=100,
        dig= 0;
      elseif max(abs(yLim(ig,:)))>=1,
        dig= 1;
      else
        dig= 2;
      end
      axesTitle{ia}= sprintf('%s  [%g %g] %s', ...
                             axesTitle{ia}, ...
                             trunc(yLim(ig,:), dig), opt.yUnit);
    end
  end
end
%warning(w_cm);

he= [];
for ia= 1:nDisps,
  ic= dispChans(ia);
  axes(H.ax(ia));
  xLim= get(gca, 'xLim');
  if strcmp(opt.axisType, 'cross'),
    shiftAwayFromBorder= 0;
  else
    shiftAwayFromBorder= 0.05;
  end    
  switch(opt.axisTitleHorizontalAlignment),
   case 'left',
    x= xLim(1) + diff(xLim)*shiftAwayFromBorder;
   case 'center',
    x= mean(xLim);
   case 'right',
    x= xLim(2) - diff(xLim)*shiftAwayFromBorder;
  end
  yl= yLim(ch2group(ia),:);
  yl_axis= yl + [1 -1]*diff(yl)*(1-1/opt.oversizePlot)/2;
  if ~isempty(strmatch(opt.axisTitleVerticalAlignment, {'bottom','baseline'})),
    yl_axis= yl_axis([2 1]);
  end
  y= yl_axis(2-strcmpi(opt.yDir, 'reverse'));
  oldUnits= get(gca, 'units');
  set(gca, 'units', 'pixel');
  pos_pixel= get(gca, 'position');
  set(gca, 'units',oldUnits);
  if strcmpi(opt.xZeroLine, 'on'),
    hl= line(xLim, [0 0], ...
             'color',opt.zeroLineColor, 'lineStyle',opt.zeroLineStyle);
    moveObjectBack(hl);
    if strcmpi(opt.axisType, 'cross'),
      xTick= get(gca, 'xTick');
      set(gca, 'xTickMode','manual');
      tlen= diff(yl)/pos_pixel(4)*opt.zeroLineTickLength;
      hl= line([xTick;xTick], [-1;1]*tlen*ones(1,length(xTick)), ...
               'color',opt.zeroLineColor);
    end
  end
  if strcmpi(opt.yZeroLine, 'on'),
    hl= line([0 0], yl_axis, ...
             'color',opt.zeroLineColor, 'lineStyle',opt.zeroLineStyle);
    moveObjectBack(hl);
    if strcmpi(opt.axisType, 'cross'),
      yTick= get(gca, 'yTick');
      yTick= yTick(find(yTick>yl_axis(1) & yTick<yl_axis(2)));
      set(gca, 'yTickMode','manual');
      tlen= diff(xLim)/pos_pixel(3)*opt.zeroLineTickLength;
      hl= line([-1;1]*tlen*ones(1,length(yTick)), [yTick;yTick], ...
               'color',opt.zeroLineColor);
    end
  end
  if isfield(epo, 'refIval'),
    yPatch= [-0.05 0.05] * diff(yl);
    if length(opt.refCol)==1,
      refCol= opt.refCol*[1 1 1];
    else
      refCol= opt.refCol;
    end
    hp= patch(epo.refIval([1 2 2 1]), yPatch([1 1 2 2]), refCol);
    set(hp, 'edgeColor','none');
    moveObjectBack(hp);
  end
  he= [he text(x, y, axesTitle{ia})];
end
set(he, 'verticalAlignment',opt.axisTitleVerticalAlignment, ...
        'horizontalAlignment',opt.axisTitleHorizontalAlignment, ...
        'color',opt.axisTitleColor, ...
        'fontWeight',opt.axisTitleFontWeight, ...
        'fontSize',opt.axisTitleFontSize);

if isfield(mnt, 'scale_box') & all(~isnan(mnt.scale_box)),
  ax_idx= chanind(mnt.clab(dispChans), opt.scaleGroup{1});
  axes(H.ax(ax_idx(1)));
  H.scale= grid_addScale(mnt, opt);
end


if ~strcmp(opt.titleDir, 'none'),
  tit= '';
  if isfield(opt, 'title'),
    tit= [opt.title ':  '];
  elseif isfield(epo, 'title'),
    tit= [untex(epo.title) ':  '];
  end
  if isfield(epo, 'className'),
    tit= [tit, vec2str(epo.className, [], ' / ') ', '];
  end
  tit= [tit, 'N= ' vec2str(nEvents,[],'/') ',  '];
  if isfield(epo, 't'),
    tit= [tit, sprintf('[%g %g] %s  ', trunc(epo.t([1 end])), opt.xUnit)];
  end
  tit= [tit, sprintf('[%g %g] %s', trunc(yLim(1,:)), opt.yUnit)];
  if strcmpi(opt.yDir, 'reverse'),
    tit= [tit, ' neg. up'];
  end
  if isfield(opt, 'titleAppendix'),
    tit= [tit, ', ' opt.titleAppendix];
  end
  H.title= addTitle(tit, opt.titleDir);
end

if ~isempty(opt.shiftAxesUp) & opt.shiftAxesUp~=0,
  shiftAxesUp(opt.shiftAxesUp);
end

if nargout==0,
  clear H;
end
