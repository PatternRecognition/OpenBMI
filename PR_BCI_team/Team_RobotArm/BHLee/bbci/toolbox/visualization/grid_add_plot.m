function grid_add_plot(epo, mnt, OPT, varargin)

if ~exist('OPT','var'),
  OPT= [];
end
if isfield(OPT, 'colorOrder') & isequal(OPT.colorOrder,'rainbow'),
  nChans= size(epo.y,1);
  OPT.colorOrder= hsv2rgb([(0.5:nChans)'/nChans ones(nChans,1)*[1 0.85]]);
end
if ~isfield(OPT, 'colorOrder') & size(epo.y,1)==1,
  OPT.colorOrder= [0 0 0];
end


dispChans= find(ismember(strhead(mnt.clab), strhead(epo.clab)));
if isfield(mnt, 'box'),
  dispChans= intersect(dispChans, find(~isnan(mnt.box(1,1:end-1))));
end

style_list= {'xGrid', 'xTick', 'xTickLabel', 'xLim', ...
             'xTickMode', 'xTickLabelMode', ...
             'yGrid', 'yTick', 'yTickLabel', 'yLim', 'yDir', ...
             'yTickMode', 'yTickLabelMode', ...
             'colorOrder', 'lineStyleOrder', ...
             'tickLength', ...
             'fontName', 'fontSize', 'fontUnits', 'fontWeight'};
ai= 0;
axesStyle= {};
opt_fields= fieldnames(OPT);
for is= 1:length(style_list),
  sm= strmatch(lower(style_list{is}), lower(opt_fields), 'exact');
  if length(sm)==1,
    ai= ai+2;
    axesStyle(ai-1:ai)= {style_list{is}, getfield(OPT, opt_fields{sm})};
  end
end

line_style_list= {'lineWidth', 'lineStyle'};
ai= 0;
lineStyle= {};
for is= 1:length(line_style_list),
  sm= strmatch(lower(line_style_list{is}), lower(opt_fields), 'exact');
  if length(sm)==1,
    ai= ai+2;
    lineStyle(ai-1:ai)= {line_style_list{is}, getfield(OPT, opt_fields{sm})};
  end
end

flags= {'grid', varargin{:}};
hsp= grid_getSubplots(mnt.clab(dispChans));
nDisps= length(dispChans);
for ia= 1:nDisps,
  ic= dispChans(ia);
  axes(hsp(ia));
  set(hsp(ia), axesStyle{:});
  hold on;      %% otherwise axis properties like colorOrder are lost
  [nEvents, hp, hc]= showERP(epo, mnt, mnt.clab{ic}, flags{:});
  if length(lineStyle)>0,
    for hpp= hp',
      set(hpp, lineStyle{:});
    end
  end
  hold off; box on;
  set(hsp(ia), axesStyle{:});
  hl= legend;
  if ~isempty(hl),
    oldPos= get(hl, 'position');
    old= getfield(get(hl, 'userData'), 'lstrings');
    new= cat(2, {epo.className{:}, 'dummy'}, old);
    hl= legend(new);
    set(hl, 'position',oldPos);
  end
end
