function h= showERPplusScalps(erp, mnt, clab, ival, varargin)

bbci_obsolete(mfilename, 'scalpEvolutionPlusChannel');

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'lineWidth',3, ...
                  'ival_color',0.85*[1 1 1], ...
                  'markIval', size(ival,1)==1, ...
                  'xGrid', 'on', ...
                  'yGrid', 'on', ...
                  'legend_pos', 0);

if size(erp.x,3)>1,
  erp= proc_average(erp);
end

[axesStyle, lineStyle]= opt_extractPlotStyles(opt);
nClasses= length(erp.className);
nIvals= size(ival,1);
if nIvals<nClasses,
  ival= repmat(ival, [nClasses, 1]);
end

clf;
for cc= 1:nClasses,
  subplotxl(1, nClasses+1, cc+1, 0.02, [0.07 0.02 0.1]);
  ot= setfield(opt, 'scalePos','none');
  ot.class= cc;
  ot= set_defaults(ot, 'linespec', {'linewidth',2, 'color',[0 0 0]});
  [h.ax_topo(cc), h.H_topo(cc)]= ...
      plotMeanScalpPattern(erp, mnt, ival(cc,:), ot);
  yLim= get(gca, 'yLim');
  h.text(cc)= text(mean(xlim), yLim(2)+0.06*diff(yLim), erp.className{cc});
  if isfield(opt, 'colorOrder'),
    set(h.text(cc), 'color',opt.colorOrder(cc,:));
  end
  ival_str= sprintf('%d - %d ms', ival(cc,:));
  h.text_ival(cc)= text(mean(xlim), yLim(1)-0.04*diff(yLim), ival_str);
  axis_aspectRatioToPosition;   %% makes colorbar appear in correct size
end
set(h.text, 'horizontalAli','center', ...
            'visible','on', ...
            'fontSize',12, 'fontWeight','bold');
set(h.text_ival, 'verticalAli','top', 'horizontalAli','center', ...
                 'visible','on');  
h.cb= colorbar_aside;
unifyCLim(h.ax_topo, [zeros(1,nClasses-1) 1]);

h.ax_erp= subplotxl(1, nClasses+1, 1, 0.06, [0.06 0.02 0.1]);
topopos= get(h.ax_topo(end), 'position');
pos= get(h.ax_erp, 'position');
pos([2 4])= topopos([2 4]);
set(h.ax_erp, 'position',pos);
if ~isempty(axesStyle),
  set(h.ax_erp, axesStyle{:});
end
hold on;   %% otherwise axis properties like colorOrder are lost
[nEvents, h.plot_erp]= showERP(erp, mnt, clab);
if length(lineStyle)>0,
  for hpp= h.plot_erp',
    set(hpp, lineStyle{:});
  end
end
set(h.ax_erp, 'box','on');
if opt.markIval,
  for cc= 1:nIvals,
    grid_markIval(ival(cc,:), clab, ...
                  opt.ival_color(min(cc,size(opt.ival_color,1)),:));
  end
end
set(get(h.ax_erp, 'title'), 'fontSize',12, 'fontWeight','bold');
h.leg= legend(erp.className, opt.legend_pos);

if nargout<1,
  clear h
end
