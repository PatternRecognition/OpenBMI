function h= showERPplusScalpSeries(erp, mnt, clab, ival, varargin)
%h= showERPplusScalpSeries(erp, mnt, clab, ival, <opts>)
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
%                    default [0.75 1 1; 1 0.75 1].
%      .legend_pos - specifies the position of the legend in the ERP plot,
%                    default 0 (see help of legend for choices).
%      the opts struct is passed to plotMeanScalpPattern
%
% OUT h - struct of handles to the created graphic objects.

%% blanker@first.fhg.de 01/2005

bbci_obsolete(mfilename, 'scalpEvolutionPlusChannel');

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'lineWidth',3, ...
                  'ival_color',[0.75 1 1; 1 0.75 1], ...
                  'legend_pos', 0);

if isfield(opt, 'colorOrder') & isequal(opt.colorOrder,'rainbow'),
  nChans= size(erp.y,1);
  opt.colorOrder= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
end

%if iscell(erp),
%  rsq= erp{2};
%  if ndim(rsq)>2,
%    error('rsq structure must not have more than one ''class''');
%  end
%  erp= erp{1};
%else
%  rsq= [];
%end
if size(erp.x,3)>1,
  erp= proc_average(erp);
end

[axesStyle, lineStyle]= opt_extractPlotStyles(opt);

if size(ival,1)==1,
  ival= [ival(1:end-1)', ival(2:end)'];
end
nIvals= size(ival,1);
nClasses= length(erp.className);

clf;
h.ax_erp= subplotxl(1+nClasses, 1, 1, 0.05, [0.07 0 0.05]);
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

nColors= size(opt.ival_color,1);
for cc= 1:nColors,
  grid_markIval(ival(cc:nColors:end,:), clab, opt.ival_color(cc,:));
end
if ~isequal(opt.legend_pos, 'none'),
  h.leg= legend(erp.className, opt.legend_pos);
end

%if ~isempty(rsq),
%  grid_addBars(rsq, 'box','on', 'vpos',1);
%end

%erp= proc_appendEpochs(erp, rsq);
%nClasses= nClasses+1;
for cc= 1:nClasses,
  for ii= 1:nIvals,
    subplotxl(1+nClasses, nIvals, ii+cc*nIvals, ...
              [0.01 0.03 0.05], [0.05 0.02 0.1]);
    ot= setfield(opt, 'scalePos','none');
    ot= setfield(ot, 'class',cc);
    ot.linespec= {'linewidth',2, ...
                  'color',opt.ival_color(mod(ii-1,nColors)+1,:)};
    h.ax_topo(cc,ii)= plotMeanScalpPattern(erp, mnt, ival(ii,:), ot);
  end
  h.cb= colorbar_aside;
  unifyCLim(h.ax_topo(cc,:), [zeros(1,nIvals-1) 1]);
  pos= get(h.ax_topo(cc,end), 'position');
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

if nargout<1,
  clear h
end
