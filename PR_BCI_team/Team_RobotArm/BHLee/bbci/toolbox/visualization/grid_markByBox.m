function H= grid_markByBox(fractiles, varargin)
%H= grid_markByBox(fractiles, <opts>)
%
% IN  fractiles - vector, e.g. [min 25%ile median 75%tile max],
%                 as obtained by 'percentiles(blah, [])'.
%     opts - struct or property/value list with optional fields/properties:
%       .clab     - channels which should be marked
%       .color    - color of the box plot
%       .linespec - linepsec of the box plot (overrides .color)
%       .height   - relative height of the box plot, default 0.075
%       .vpos     - vertical position of the box plot (0: bottom, 1: top)
% OUT
%   H - handle to graphic opbjects
%
% EXAMPLE
%  grid_plot(erp, mnt, defopt_erps);
%  grid_markByBox(percentiles(mrk.latency, [5 25 50 75 95]);

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'clab', [], ...
                  'color', 0.3*[1 1 1], ...
                  'linespec', {}, ...
                  'height', 0.05, ...
                  'vpos',0);

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
  H(ii).median= line(fractiles([3 3]'), [yy], opt.linespec{:});
  H(ii).whisker_ends= line(fractiles([1 1; 5 5]'), [yy, yy], opt.linespec{:})';
  H(ii).whisker= line(fractiles([1 2; 4 5]'), [1 1; 1 1]*mean(yy), ...
                      opt.linespec{:})';
  H(ii).box= line(fractiles([2 4 4 2 2]), yy([1 1 2 2 1]), opt.linespec{:});
end
moveObjectBack(struct2array(H));
axes(old_ax);
