function hnd= plotPatternsPlusSpec(epo, mnt, W, varargin)
%hnd= plotPatternsPlusSpec(epo, mnt, W, <OPT>)
%
% W may, e.g., be the unmixing matrix of an ICA.

OPT= propertylist2struct(varargin{:});
OPT= set_defaults(OPT, ...
                  'nRows', 5, ...
                  'nCols', 3, ...
                  'frac', [0.36], ...
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
spec_axesStyle= {'box','on'};
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
    if cc==1,
      clf;
    else
      figure;
    end
    pp= 0;
  end
  pp= pp+1;
  iv= m - 1 - floor((pp-1)/n);
  ih= mod(pp-1, n);

  pat_pos= [mh(1)+ih*(mh(2)+ph), mv(1)+iv*(mv(2)+pv), ph*OPT.frac, pv];
  spec_pos= [mh(1)+ih*(mh(2)+ph)+ph*OPT.frac, mv(1)+iv*(mv(2)+pv), ...
             ph*(1-OPT.frac), pv];
  
  hnd.ax_pat(cc)= axes('position', pat_pos);
  scalpPlot(mnt, A(:,ic), OPT);

  hnd.ax_spec(cc)= axes('position', spec_pos);
  set(hnd.ax_spec(cc), spec_axesStyle{:});
  hold on;   %% otherwise axis properties like colorOrder get lost
  [dmy, hnd.plot(:,cc)]= showERP(spec, mnt, ic);
  hold off;
  set(hnd.ax_spec(cc), 'xTickLabel',[], 'yTickLabel',[], ...
                    'xLimMode','manual', 'yLimMode','manual', ...
                    spec_axesStyle{:});
  if isfield(opt_spec,'yZeroLine') & strcmpi(opt_spec.yZeroLine, 'on'),
    hyzl= line(get(hnd.ax_spec(cc),'xLim'), [0 0]);
    set(hyzl, 'color','k');
    moveObjectBack(hyzl);
  end
  tit= sprintf('ic%d  [%g %g]', ic, trunc(get(hnd.ax_spec(cc),'yLim')));
  hnd.title(cc)= title(tit, 'verticalAlign','top', 'fontWeight','demi', ...
                       'unit','normalized');
  pos= get(hnd.title(cc), 'position');
  pos(2)= 1;
  set(hnd.title(cc), 'position',pos, 'string',tit);

  drawnow;
end
