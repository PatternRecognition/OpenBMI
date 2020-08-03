function h= scalpEvolutionPlusChannel(erp, mnt, clab, ival, varargin)
%SCALPEVOLUTIONPLUSCHANNEL - Display evolution of scalp topographies 
%
%Usage:
% H= scalpEvolutionPlusChannel(ERP, MNT, CLAB, IVAL, <OPTS>)
%
% Makes an ERP plot in the upper panel with given interval marked,
% and draws below scalp topographies for all marked intervals,
% separately for each each class. For each classes topographies are
% plotted in one row and shared the same color map scaling. (In future
% versions there might be different options for color scaling.)
%
% IN: erp  - struct of epoched EEG data. For convenience used classwise
%            averaged data, e.g., the result of proc_average.
%     mnt  - struct defining an electrode montage
%     clab - label of the channel(s) which are to be displayed in the
%            ERP plot.
%     ival - [nIvals x 2]-sized array of interval, which are marked in the
%            ERP plot and for which scalp topographies are drawn.
%            When all interval are consequtive, ival can also be a
%            vector of interval borders.
%     opts - struct or property/value list of optional fields/properties:
%      .ival_color - [nColors x 3]-sized array of rgb-coded colors
%                    with are used to mark intervals and corresponding 
%                    scalps. Colors are cycled, i.e., there need not be
%                    as many colors as interval. Two are enough,
%                    default [0.4 1 1; 1 0.6 1].
%      .legend_pos - specifies the position of the legend in the ERP plot,
%                    default 0 (see help of legend for choices).
%      .printIvalUnits - appends the unit when writing the ival borders,
%                    default 1
%      the opts struct is passed to scalpPattern
%
% OUT h - struct of handles to the created graphic objects.
%
%See also scalpEvolution, scalpPatternsPlusChannel, scalpPlot.

% Author(s): Benjamin Blankertz, Jan 2005

fig_visible = strcmp(get(gcf,'Visible'),'on'); % If figure is already invisible jvm_* functions should not be called
if fig_visible
  jvm= jvm_hideFig;
end

opt= propertylist2struct(varargin{:});
[opt_orig, isdefault]= ...
    set_defaults(opt, ...
                 'lineWidth',3, ...
                 'ival_color',[0.4 1 1; 1 0.6 1], ...
                 'xUnit', '[ms]', ...
                 'yUnit', '[\muV]', ...
                 'printIval', isempty(clab), ...
                 'printIvalUnits', 1, ...
                 'globalCLim', 0, ...
                 'scalePos', 'vert', ...
                 'shrinkColorbar', 0, ...
                 'plotChannel', ~isempty(clab), ...
                 'channelAtBottom', 0, ...
                 'subplot', [], ...
                 'subplot_channel', [], ...
                 'figure_color', [1 1 1], ...
                 'legend_pos', 'Best');
opt= setfield(opt_orig, 'fig_hidden', 1);

if isfield(erp, 'xUnit'),
  [opt,isdefault]= opt_overrideIfDefault(opt, isdefault, ...
                                         'xUnit', erp.xUnit);
end

if isfield(erp, 'yUnit'),
  [opt,isdefault]= opt_overrideIfDefault(opt, isdefault, ...
                                         'yUnit', erp.yUnit);
end

if isfield(opt, 'colorOrder'),
  if isequal(opt.colorOrder,'rainbow'),
    nChans= size(erp.y,1);
    opt.colorOrder= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
%  else
%    if size(opt.colorOrder,1)<size(erp.y,1),
%      opt.colorOrder= repmat(opt.colorOrder, [size(erp.y,1) 1]);
%    end
  end
else
  opt.colorOrder= get(gca, 'ColorOrder');
end

if max(sum(erp.y,2))>1,
  erp= proc_average(erp);
end

[axesStyle, lineStyle]= opt_extractPlotStyles(opt);

if size(ival,1)==1,
  ival= [ival(1:end-1)', ival(2:end)'];
end
if any(ival(:)>erp.t(end)),
  warning('interval out of epoch range: truncating');
  ival= min(ival, erp.t(end));
end
if any(ival(:)<erp.t(1)),
  warning('interval out of epoch range: truncating');
  ival= max(ival, erp.t(1));
end

nIvals= size(ival,1);
nColors= size(opt.ival_color,1);
nClasses= length(erp.className);

if isempty(opt.subplot),
  clf;
end
set(gcf, 'color',opt.figure_color);

subplot_offset= 0;
if opt.plotChannel,
  if ~isempty(opt.subplot_channel),
    h.ax_erp= opt.subplot_channel;
    backaxes(h.ax_erp);
  else
    if opt.channelAtBottom,
      h.ax_erp= subplotxl(1+nClasses, 1, 1+nClasses, ...
                          [0.1 0.05 0.1], [0.09 0 0.05]);
    else
      h.ax_erp= subplotxl(1+nClasses, 1, 1, 0.05, [0.09 0 0.05]);
      subplot_offset= 1;
    end
  end
  if ~isempty(axesStyle),
    set(h.ax_erp, axesStyle{:});
  end
  hold on;   %% otherwise axis properties like colorOrder are lost
  h.channel= plotChannel(erp, clab, opt, 'legend',0);
  for cc= 1:min(nColors,size(ival,1)),
    grid_markIval(ival(cc:nColors:end,:), clab, opt.ival_color(cc,:));
  end
  axis_redrawFrame(h.ax_erp);
  if ~isequal(opt.legend_pos, 'none'),
    if iscell(h.channel),
      hhh= h.channel{1};
    else
      hhh= h.channel;
    end
    % Check matlab version for downward compatability
    if str2double(strtok(version,'.'))<7
      h.leg= legend(hhh.plot, erp.className, opt.legend_pos);
    else
      % check if legend_pos is integer
      if isnumeric(opt.legend_pos)
        switch(opt.legend_pos)
          case -1, loc = 'NorthEastOutside';
          case 0,  loc = 'Best';
          case 1,  loc = 'NorthEast';
          case 2,  loc = 'NorthWest';
          case 3,  loc = 'SouthWest';
          case 4,  loc = 'SouthEast';
          otherwise, loc = 'Best';
        end
        warning('Location numbers are obsolete, use "%s" instead of "%d"', ...
          loc,opt.legend_pos);
      else
        loc = opt.legend_pos;
      end
      h.leg= legend(hhh.plot, erp.className, 'Location',loc);
    end
  end
end

%if ~isempty(opt.subplot),
%  opt.subplot= reshape(opt.subplot, [nClasses, nIvals]);
%end
cb_per_ival= strcmp(opt.scalePos, 'horiz');
for cc= 1:nClasses,
  if ~any(any(erp.x(:,:,cc))),
    bbci_warning('empty_class', sprintf('class %d is empty', cc));
    continue;
  end
  for ii= 1:nIvals,
    if any(isnan(ival(ii,:))),
      continue;
    end
    if ~isempty(opt.subplot),
      backaxes(opt.subplot(cc, ii));
    else
      subplotxl(nClasses+opt.plotChannel, nIvals, ...
                ii+(cc-1+subplot_offset)*nIvals, ...
                [0.01+0.08*cb_per_ival 0.03 0.05], [0.05 0.02 0.1]);
    end
    ot= setfield(opt, 'scalePos','none');
    ot= setfield(ot, 'class',cc);
    ot.linespec= {'linewidth',2, ...
                  'color',opt.ival_color(mod(ii-1,nColors)+1,:)};
    h.scalp(cc,ii)= scalpPattern(erp, mnt, ival(ii,:), ot);
    if cc==nClasses 
      if opt.printIval,
        yLim= get(gca, 'yLim');
        if opt.printIvalUnits==2,
          ival_str= sprintf('%s - %s\n%s', num2str(ival(ii,1)), num2str(ival(ii,2)), opt.xUnit);
        elseif opt.printIvalUnits==1,
          ival_str= sprintf('%s - %s %s', num2str(ival(ii,1)), num2str(ival(ii,2)), opt.xUnit);
        else
          ival_str= sprintf('%s - %s', num2str(ival(ii,1)), num2str(ival(ii,2)));
        end;
        h.text_ival(ii)= text(mean(xlim), yLim(1)-0.04*diff(yLim), ival_str, ...
                              'verticalAli','top', 'horizontalAli','center');
      end
      if cb_per_ival,
        h.cb(ii)= colorbar_aside('horiz');
        if ~opt.globalCLim,
%          unifyCLim([h.scalp(:,ii).ax], [zeros(1,nClasses-1) h.cb(ii)]);
          unifyCLim([h.scalp(:,ii).ax]);
        end
      end
    end
  end
  if strcmp(opt.scalePos, 'vert'),
    h.cb(cc)= colorbar_aside;
    ylabel(h.cb(cc), opt.yUnit);
    if opt.shrinkColorbar>0,
      cbpos= get(h.cb(cc), 'Position');
      cbpos(2)= cbpos(2) + cbpos(4)*opt.shrinkColorbar/2;
      cbpos(4)= cbpos(4) - cbpos(4)*opt.shrinkColorbar;
      set(h.cb(cc), 'Position',cbpos);
    end
    if ~opt.globalCLim,
%      unifyCLim([h.scalp(cc,:).ax], [zeros(1,nIvals-1) h.cb(cc)]);
      unifyCLim([h.scalp(cc,:).ax]);
    end
  end
  pos= get(h.scalp(cc,end).ax, 'position');
  yy= pos(2)+0.5*pos(4);
  h.background= getBackgroundAxis;
  h.text(cc)= text(0.01, yy, erp.className{cc});
  set(h.text(cc), 'verticalAli','top', ...
                  'horizontalAli','center', ...
                  'rotation',90, ...
                  'visible','on', ...
                  'fontSize',12, ...
                  'fontWeight','bold');
  if isfield(opt, 'colorOrder'),
    ccc= 1+mod(cc-1, size(opt.colorOrder,1));
    set(h.text(cc), 'color',opt.colorOrder(ccc,:));
  end
end
if opt.globalCLim,
%  ucb= [zeros(nClasses, nIvals-1) ones(nClasses,1)];
%  unifyCLim([h.scalp.ax], isfield(h, 'cb'));
  unifyCLim([h.scalp.ax]);
end

if nargout<1,
  clear h
end

if fig_visible
  jvm_restoreFig(jvm, opt_orig);
end
