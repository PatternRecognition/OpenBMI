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
%      the opts struct is passed to scalpPattern
%
% OUT h - struct of handles to the created graphic objects.
%
%See also scalpEvolution, scalpPatternsPlusChannel, scalpPlot.

% Author(s): Benjamin Blankertz, Jan 2005

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'lineWidth',3, ...
                 'ival_color',[0.4 1 1; 1 0.6 1], ...
                 'xUnit', 'ms', ...
                 'printIval', isempty(clab), ...
                 'globalCLim', 0, ...
                 'scalePos', 'vert', ...
                 'plotChannel', ~isempty(clab), ...
                 'figure_color', [1 1 1], ...
                 'legend_pos', 0);

if isfield(erp, 'xUnit'),
  [opt,isdefault]= opt_overrideIfDefault(opt, isdefault, ...
                                         'xUnit', erp.xUnit);
end

if isfield(opt, 'colorOrder'),
  if isequal(opt.colorOrder,'rainbow'),
    nChans= size(erp.y,1);
    opt.colorOrder= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
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
nIvals= size(ival,1);
nColors= size(opt.ival_color,1);
nClasses= length(erp.className);

clf;
set(gcf, 'color',opt.figure_color);
if opt.plotChannel,
  h.ax_erp= subplotxl(1+nClasses, 1, 1, 0.05, [0.09 0 0.05]);
  if ~isempty(axesStyle),
    set(h.ax_erp, axesStyle{:});
  end
  hold on;   %% otherwise axis properties like colorOrder are lost
  H.channel= plotChannel(erp, clab, opt, 'legend',0);
  for cc= 1:nColors,
    grid_markIval(ival(cc:nColors:end,:), clab, opt.ival_color(cc,:));
  end
  axis_redrawFrame(h.ax_erp);
  if ~isequal(opt.legend_pos, 'none'),
    if iscell(H.channel),
      hhh= H.channel{1};
    else
      hhh= H.channel;
    end
    h.leg= legend(hhh.plot, erp.className, opt.legend_pos);
  end
end
cb_per_ival= strcmp(opt.scalePos, 'horiz');
for cc= 1:nClasses,
  for ii= 1:nIvals,
    subplotxl(nClasses+opt.plotChannel, nIvals, ...
              ii+(cc-1+opt.plotChannel)*nIvals, ...
              [0.01+0.08*cb_per_ival 0.03 0.05], [0.05 0.02 0.1]);
    ot= setfield(opt, 'scalePos','none');
    %ot= setfield(ot, 'class',cc);
    ot.linespec= {'linewidth',2, ...
                  'color',opt.ival_color(mod(ii-1,nColors)+1,:)};
    h.scalp(cc,ii)= scalpPattern(erp, mnt, ival(ii,:), ot);
    if cc==nClasses 
      if opt.printIval,
        yLim= get(gca, 'yLim');
        ival_str= sprintf('%d - %d %s', ival(ii,:), opt.xUnit);
        h.text_ival(ii)= text(mean(xlim), yLim(1)-0.04*diff(yLim), ival_str, ...
                              'verticalAli','top', 'horizontalAli','center');
      end
      if cb_per_ival,
        h.cb(ii)= colorbar_aside('horiz');
        if ~opt.globalCLim,
          unifyCLim([h.scalp(:,ii).ax], [zeros(1,nClasses-1) h.cb(ii)]);
        end
      end
    end
  end
  if strcmp(opt.scalePos, 'vert'),
    h.cb(cc)= colorbar_aside;
    if ~opt.globalCLim,
      unifyCLim([h.scalp(cc,:).ax], [zeros(1,nIvals-1) h.cb(cc)]);
    end
  end
  pos= get(h.scalp(cc,end).ax, 'position');
  yy= pos(2)+0.5*pos(4);
  h.background= getBackgroundAxis;
  h.text= text(0.01, yy, erp.className{cc});
  set(h.text, 'verticalAli','top', 'horizontalAli','center', ...
              'rotation',90, 'visible','on', ...
              'fontSize',12, 'fontWeight','bold');
  if isfield(opt, 'colorOrder'),
    set(h.text, 'color',opt.colorOrder(cc,:));
  end
end
if opt.globalCLim,
%  ucb= [zeros(nClasses, nIvals-1) ones(nClasses,1)];
  unifyCLim([h.scalp.ax], 1);
end

if nargout<1,
  clear h
end
