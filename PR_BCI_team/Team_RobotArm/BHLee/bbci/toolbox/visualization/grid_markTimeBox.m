function H= grid_markTimeBox(intervals, varargin)
%H= grid_markTimeBox(intervals, <opts>)
%
% IN  time intervals - vectors, e.g. [min max],
%     opts - struct or property/value list with optional fields/properties:
%       .clab     - channels which should be marked
%       .color    - colors of the line
%       .linespec - linepsec of the box plot (overrides .color)
%       .height   - relative height of the line
%       .vpos     - vertical position of the line
% OUT
%   H - handle to graphic opbjects

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'clab', [], ...
                  'color', 0.3*[1 1 1], ...
                  'linespec', {}, ...
                  'height', 0.075, ...
                  'vpos',0, ...
                  'move_to_background', 1);

if isempty(opt.linespec),
  opt.linespec= {'color',opt.color};
end

old_ax= gca;
hsp= grid_getSubplots(opt.clab);
for ii= 1:length(hsp),
  ih= hsp(ii);
  axes(ih);
  yl= get(ih, 'yLim');
  yh= opt.height*diff(yl);
  delta= 0.005;
  y_lower= yl(1) + (opt.vpos+delta)*(diff(yl)-yh-delta);
  yy= [y_lower; y_lower+yh];
  for jj = 1:size(intervals, 1),
      H(ii,jj).box= line(intervals(jj,[1 2]), yy([1 1]), opt.linespec{:});
  end
end
if opt.move_to_background,
  moveObjectBack(struct2array(H));
end
axes(old_ax);
