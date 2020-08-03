function h= scalpLoadingsPlusChannel(erp, mnt, clab, ival, varargin)
%SCALPLOADINGSPLUSCHANNEL - Display classwise topographies and one channel
%
%Usage:
% H= scalpPatternsPlusChannel(ERP, MNT, CLAB, IVAL, <OPTS>)
%
%Input:
% ERP  - struct of epoched EEG data. For convenience used classwise
%        averaged data, e.g., the result of proc_average.
% MNT  - struct defining an electrode montage
% CLAB - label of the channel(s) which are to be displayed in the
%        ERP plot.
% IVAL - The time interval for which scalp topographies are to be plotted.
%        May be either one interval for all classes, or specific
%        intervals for each class. In the latter case the k-th row of IVAL
%        defines the interval for the k-th class.
% OPTS - struct or property/value list of optional fields/properties:
%  .legend_pos - specifies the position of the legend in the ERP plot,
%                default 0 (see help of legend for choices).
%  .markIval - When true, the time interval is marked in the channel plot.
%  The opts struct is passed to scalpPattern.
%
%Output:
% H: Handle to several graphical objects.
%
%See also scalpPatterns, scalpEvolutionPlusChannel, scalpPlot.

% Author(s): Benjamin Blankertz, Jan 2005

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'lineWidth',3, ...
                 'ival_color',0.85*[1 1 1], ...
                 'markIval', size(ival,1)==1, ...
                 'xGrid', 'on', ...
                 'yGrid', 'on', ...
                 'xUnit', '[ms]', ...
                 'plotChannel', ~isempty(clab), ...
                 'scalePos', 'vert', ...
                 'legend_pos', 0, ...
                 'subplot', []);

if isempty(ival),
  ival= erp.t([1 end]);
elseif size(ival,2)==1,
  ival= [ival ival];
elseif size(ival,2)>2,
  error('IVAL must be sized [N 2]');
end

if isfield(erp, 'xUnit'),
  [opt,isdefault]= opt_overrideIfDefault(opt, isdefault, 'xUnit', erp.xUnit);
end

if max(sum(erp.y,2))>1,
  erp= proc_average(erp);
end

if ~isfield(opt, 'colAx') & all(erp.x>=0),
  opt.colAx= 'range';
  warning('automatically set opt.colAx= ''range''.');
end

[axesStyle, lineStyle]= opt_extractPlotStyles(opt);
nClasses= length(erp.className);
nIvals= size(ival,1);
if nIvals<nClasses,
  ival= repmat(ival, [nClasses, 1]);
end

if isempty(opt.subplot),
  clf;
end
for cc= 1:nClasses,
  if isempty(opt.subplot),
    subplotxl(1, nClasses+opt.plotChannel, cc+opt.plotChannel, ...
              0.02, [0.07 0.02 0.1]);
  else
    axes(opt.subplot(cc));
  end
  ot= setfield(opt, 'scalePos','none');
  ot.class= cc;
  ot= set_defaults(ot, 'linespec', {'linewidth',2, 'color',[0 0 0]});
  h.H_topo(cc)= scalpLoading(erp, mnt, ival(cc,:), ot);
  yLim= get(gca, 'yLim');
  h.text(cc)= text(mean(xlim), yLim(2)+0.06*diff(yLim), erp.className{cc});
  if isfield(opt, 'colorOrder'),
    set(h.text(cc), 'color',opt.colorOrder(cc,:));
  end
  ival_str= sprintf('%g - %g %s', ival(cc,:), opt.xUnit);
  h.text_ival(cc)= text(mean(xlim), yLim(1)-0.04*diff(yLim), ival_str);
%  axis_aspectRatioToPosition;   %% makes colorbar appear in correct size
end
set(h.text, 'horizontalAli','center', ...
            'visible','on', ...
            'fontSize',12, 'fontWeight','bold');
set(h.text_ival, 'verticalAli','top', 'horizontalAli','center', ...
                 'visible','on');  
if ismember(opt.scalePos, {'horiz','vert'}),
  h.cb= colorbar_aside(opt.scalePos);
  %% hack to fix a matlab bug
  ud= get(h.cb, 'UserData');
  ud.orientation= opt.scalePos;
  set(h.cb, 'UserData',ud);
  %% put yUnit on top of colorbar
  if isfield(erp, 'yUnit'),
    axes(h.cb);
    yl= get(h.cb, 'YLim');
    h.yUnit= text(mean(xlim), yl(2), erp.yUnit);
    set(h.yUnit, 'horizontalAli','center', 'verticalAli','bottom');
  end
  unifyCLim([h.H_topo.ax], [zeros(1,nClasses-1) h.cb]);
elseif nClasses>1,
  unifyCLim([h.H_topo.ax], [zeros(1,nClasses)]);
end

if opt.plotChannel,
  if isempty(opt.subplot),
    h.ax_erp= subplotxl(1, nClasses+1, 1, 0.06, [0.06 0.02 0.1]);
  else
    h.ax_erp= opt.subplot(nClasses+1);
    axes(h.ax_erp);
  end
  topopos= get(h.H_topo(end).ax, 'position');
  pos= get(h.ax_erp, 'position');
  pos([2 4])= topopos([2 4]);
  set(h.ax_erp, 'position',pos);
  if ~isempty(axesStyle),
    set(h.ax_erp, axesStyle{:});
  end
  hold on;   %% otherwise axis properties like colorOrder are lost
  H.channel= plotChannel(erp, clab, opt, 'legend',0);
  if opt.markIval,
    for cc= 1:nIvals,
      grid_markIval(ival(cc,:), clab, ...
                    opt.ival_color(min(cc,size(opt.ival_color,1)),:));
    end
    axis_redrawFrame(h.ax_erp);
  end
  set(get(h.ax_erp, 'title'), 'fontSize',12, 'fontWeight','bold');
  if ~isequal(opt.legend_pos, 'none'),
    h.leg= legend(erp.className, opt.legend_pos);
  end
end

if nargout<1,
  clear h
end
