function plotPatternsPlusSpecERP(epo, mnt, W, varargin)
%plotPatternsPlusSpecERP(epo, mnt, W, <OPT>)
%
% W may, e.g., be the unmixing matrix of an ICA.

dd= 0.02;
OPT= propertylist2struct(varargin{:});
OPT= set_defaults(OPT, ...
                  'nRows', 5, ...
                  'nCols', 2, ...
                  'frac', [0.24], ...
                  'spec_ival', epo.t([1 end]), ...
                  'spec_band', [3 50], ...
                  'colAx', 'sym', ...
                  'scalePos', 'none', ...
                  'selection', 1:size(W,1));
if isfield(OPT, 'spec_opt'), 
  opt_spec= OPT.spec_opt;
else
  opt_spec= OPT;
end
if length(OPT.frac)==1,
  OPT.frac= [OPT.frac, (1-OPT.frac-2*dd)/2];
end

A= inv(W);
D= sum(A);
W= diag(D)*W;

epo= proc_linearDerivation(epo, W');
erp= proc_average(epo);
if isfield(OPT, 'erp_proc'),
  eval(OPT.erp_proc);
end
spec= proc_selectIval(epo, OPT.spec_ival);
spec= proc_spectrum(spec, OPT.spec_band);
spec= proc_average(spec);

nPats= length(OPT.selection);
m= OPT.nRows;
n= OPT.nCols;
f= [OPT.frac(1), OPT.frac(2), 1-2*dd-OPT.frac(1)-OPT.frac(2)];

mv= [0.01 0.03 0.01];
mh= [0.01 0.02 0.01];

pv= ( 0.999 - mv(1) - mv(3) - mv(2)*(m-1) ) / m;
ph= ( 0.999 - mh(1) - mh(3) - mh(2)*(n-1) ) / n;


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
ai= 0;
spec_axesStyle= {};
OPT_fields= fieldnames(opt_spec);
for is= 1:length(style_list),
  sm= strmatch(lower(style_list{is}), lower(OPT_fields), 'exact');
  if length(sm)==1,
    ai= ai+2;
    spec_axesStyle(ai-1:ai)= {style_list{is}, ...
                    getfield(opt_spec, OPT_fields{sm})};
  end
end

for cc= 1:nPats,
  ic= OPT.selection(cc);
  ff= ceil(cc/m/n);
  if mod(cc, m*n)==1,
    nicefig(ff);
    pp= 0;
  end
  pp= pp+1;
  iv= m - 1 - floor((pp-1)/n);
  ih= mod(pp-1, n);

  pat_pos= [mh(1)+ih*(mh(2)+ph), mv(1)+iv*(mv(2)+pv), ph*f(1), pv];
  erp_pos= [mh(1)+ih*(mh(2)+ph)+ph*(f(1)+dd), mv(1)+iv*(mv(2)+pv), ...
            ph*f(2), pv];
  spec_pos= [mh(1)+ih*(mh(2)+ph)+ph*(f(1)+f(2)+2*dd), mv(1)+iv*(mv(2)+pv), ...
             ph*f(3), pv];
  
  ax= axes('position', pat_pos);
  plotScalpPattern(mnt, A(:,ic), OPT);

  ax= axes('position', erp_pos);
  set(ax, axesStyle{:});
  hold on;   %% otherwise axis properties like colorOrder get lost
  if cc==nPats & pp<m*n,
    [d,d,hz,hleg]= showERP(erp, mnt, ic, 'legend');
    iv= m - 1 - floor(pp/n);
    ih= mod(pp, n);
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

  ax= axes('position', spec_pos);
  set(ax, spec_axesStyle{:});
  hold on;   %% otherwise axis properties like colorOrder get lost
  showERP(spec, mnt, ic);
  hold off;
  set(ax, 'xTickLabel',[], 'yTickLabel',[], ...
          'xLimMode','manual', 'yLimMode','manual', ...
          spec_axesStyle{:});
  if isfield(opt_spec,'yZeroLine') & strcmpi(opt_spec.yZeroLine, 'on'),
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
