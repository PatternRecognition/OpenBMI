function plotPatternsPlusERP(epo, mnt, W, varargin)
%plotPatternsPlusERP(epo, mnt, W, <OPT>)
%
% W may, e.g., be the unmixing matrix of an ICA.

%% this was the old syntax:
%plotPatternsPlusERP(epo, mnt, W, <OPT, nRows, nCols, frac>)

OPT= propertylist2struct(varargin{:});
OPT= set_defaults(OPT, ...
                  'nRows', 5, ...
                  'nCols', 3, ...
                  'frac', [0.36], ...
                  'colAx', 'sym', ...
                  'scalePos', 'none', ...
                  'selection', 1:size(W,1));

A= inv(W);
D= sum(A);
W= diag(D)*W;

erp= proc_linearDerivation(epo, W');
erp= proc_average(erp);

nPats= length(OPT.selection);

mv= [0.01 0.03 0.01];
mh= [0.01 0.02 0.01];

pv= ( 0.999 - mv(1) - mv(3) - mv(2)*(OPT.nRows-1) ) / OPT.nRows;
ph= ( 0.999 - mh(1) - mh(3) - mh(2)*(OPT.nCols-1) ) / OPT.nCols;


style_list= {'xGrid', 'xTick', 'xTickLabel', 'xLim', ...
             'xTickMode', 'xTickLabelMode', 'xColor', ...
             'yGrid', 'yTick', 'yTickLabel', 'yLim', 'yDir', ...
             'yTickMode', 'yTickLabelMode', 'yColor', ...
             'colorOrder', 'lineStyleOrder', ...
             'tickLength', 'box', ...
             'fontName', 'fontSize', 'fontUnits', 'fontWeight'};
ai= 0;
axesStyle= {};
OPT_fields= fieldnames(OPT);
for is= 1:length(style_list),
  sm= strmatch(lower(style_list{is}), lower(OPT_fields), 'exact');
  if length(sm)==1,
    ai= ai+2;
    axesStyle(ai-1:ai)= {style_list{is}, getfield(OPT, OPT_fields{sm})};
  end
end
    
for cc= 1:nPats,
  ic= OPT.selection(cc);
  ff= ceil(cc/OPT.nRows/OPT.nCols);
  if mod(cc, OPT.nRows*OPT.nCols)==1,
    nicefig(ff);
    pp= 0;
  end
  pp= pp+1;
  iv= OPT.nRows - 1 - floor((pp-1)/OPT.nCols);
  ih= mod(pp-1, OPT.nCols);

  pat_pos= [mh(1)+ih*(mh(2)+ph), mv(1)+iv*(mv(2)+pv), ph*OPT.frac, pv];
  erp_pos= [mh(1)+ih*(mh(2)+ph)+ph*OPT.frac, mv(1)+iv*(mv(2)+pv), ...
            ph*(1-OPT.frac), pv];
  
  ax= axes('position', pat_pos);
  plotScalpPattern(mnt, A(:,ic), OPT);
  
  ax= axes('position', erp_pos);
  set(ax, axesStyle{:});
  hold on;   %% otherwise axis properties like colorOrder get lost
  if cc==nPats & pp<OPT.nRows*OPT.nCols,
    [d,d,hz,hleg]= showERP(erp, mnt, ic, 'legend');
    iv= OPT.nRows - 1 - floor(pp/OPT.nCols);
    ih= mod(pp, OPT.nCols);
    leg_pos= [mh(1)+ih*(mh(2)+ph), mv(1)+iv*(mv(2)+pv), ph, pv];
    set(hleg, 'position',leg_pos);
    drawnow;
  else
    showERP(erp, mnt, ic);
  end
  hold off;
  set(ax, 'xTickLabel',[], 'yTickLabel',[], ...
          'xLimMode','manual', 'yLimMode','manual', ...
          axesStyle{:});
  if isfield(OPT,'yZeroLine') & strcmpi(OPT.yZeroLine, 'on'),
    hyzl= line(get(ax,'xLim'), [0 0]);
    set(hyzl, 'color','k');
    moveObjectBack(hyzl);
  end
  tit= sprintf('ic%d  [%g %g]', ic, trunc(get(ax,'yLim')));
  ht= title(tit, 'verticalAlign','top', 'fontWeight','demi', ...
            'unit','normalized');
  pos= get(ht, 'position');
  pos(2)= 1;
  set(ht, 'position',pos, 'string',tit);
  drawnow;
end
