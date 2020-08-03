function ht= stimutil_showD2cue(letter, linePos, varargin)
%H_D2= showD2(LETTER, LINEPOS, <OPT>)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'hpos', 0, ...
                  'vpos', 0, ...
                  'd2_fontsize', 0.3, ...
                  'd2_dist_between_bars', 0.2, ...
                  'd2_dist_to_upper_bars', 2.15, ...
                  'd2_dist_to_lower_bars', 0.15, ...
                  'handle_d2', []);
s = opt.d2_fontsize;
h = opt.hpos;
v = opt.vpos;
d = opt.d2_dist_between_bars;
u = opt.d2_dist_to_upper_bars;
l = opt.d2_dist_to_lower_bars;
par= {'HorizontalAlignment','center', ...
      'VerticalAlignment','middle',...
      'FontUnits','normalized', ...
      'FontSize',s, ...
      'Interpreter','none'};
barpos= v + s*[u+d u -l -l-d];
barchar= {' ','_'};
if ~isempty(opt.handle_d2),
  set(opt.handle_d2(1), 'String',letter);
  for li= 1:length(barpos),
    set(opt.handle_d2(1+li), 'String',barchar{1+linePos(li)});
  end
  set(opt.handle_d2, 'Visible','on');
else
  opt.handle_d2= zeros(1+length(barpos),1);
  opt.handle_d2(1)= text(h, v, letter);
  for li= 1:length(barpos),
    opt.handle_d2(1+li)= text(h, barpos(li), barchar{1+linePos(li)});
  end
  set(opt.handle_d2, par{:});
end

if nargout>0,
  ht= opt.handle_d2;
end
