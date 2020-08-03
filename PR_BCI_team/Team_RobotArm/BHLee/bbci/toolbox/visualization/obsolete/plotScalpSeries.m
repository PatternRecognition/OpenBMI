function h= plotScalpSeries(erp, mnt, ival, varargin)
%h= plotScalpSeries(erp, mnt, ival, <opts>)
%
% Draws scalp topographies for specified intervals,
% separately for each each class. For each classes topographies are
% plotted in one row and shared the same color map scaling. (In future
% versions there might be different options for color scaling.)
%
% IN: erp  - struct of epoched EEG data. For convenience used classwise
%            averaged data, e.g., the result of proc_average.
%     mnt  - struct defining an electrode montage
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

bbci_obsolete(mfilename, 'scalpEvolution');

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'lineWidth',3, ...
                  'ival_color',[0.75 1 1; 1 0.75 1], ...
                  'legend_pos', 0);

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
for cc= 1:nClasses,
  for ii= 1:nIvals,
    subplotxl(nClasses, nIvals, ii+(cc-1)*nIvals, ...
              [0.01 0.03 0.05], [0.05 0.02 0.1]);
    ot= setfield(opt, 'scalePos','none');
    ot= setfield(ot, 'class',cc);
    ot.linespec= {'linewidth',2, ...
                  'color',opt.ival_color(mod(ii-1,nColors)+1,:)};
    h.ax_topo(cc,ii)= plotMeanScalpPattern(erp, mnt, ival(ii,:), ot);
  end
%  axis_aspectRatioToPosition;   %% makes colorbar appear in correct size
  h.cb= colorbar_aside;
  unifyCLim(h.ax_topo(cc,:), [zeros(1,nIvals-1) 1]);
  pos= get(h.ax_topo(cc,end), 'position');
  h.background= getBackgroundAxis;
  h.text= text(0.01, pos(2)+0.5*pos(4), erp.className{cc});
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
