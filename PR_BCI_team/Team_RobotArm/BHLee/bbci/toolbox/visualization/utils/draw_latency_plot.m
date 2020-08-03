function H= draw_latency_plot(lat, varargin)

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                  'x', [], ...
                  'ylim', [], ...
                  'blank', [], ...
                  'blank_color', 0.7*[1 1 1], ...
                  'opt_plot', {}, ...
                  'col', [1 0 0], ...
                  'pale_factor', 0.5, ...
                  'marker', '.', ...
                  'plot_density', 0, ...
                  'plot_ma', 1, ...
                  'ma_N', 11, ...
                  'ma_win', 'sin');

if isempty(lat),
  return;
end

if opt.plot_density && isdefault.plot_ma,
  opt.plot_ma= 0;
end

if isdefault.opt_plot,
  opt.opt_plot= {'Color',opt.col};
end

if ~isequal(opt.ma_win, 'sin'),
  error('requested ma_win not implemented');
end

win= sin([1:opt.ma_N]'/(opt.ma_N+1)*pi);
if isempty(opt.x),
  opt.x= 1:length(lat);
end

H.ax= gca;
set(H.ax, 'NextPlot','add');
ud= get(H.ax, 'UserData');
if opt.plot_ma,
  lat_ma= movingAverage(lat', opt.ma_N, ...
                        'method','centered', ...
                        'window', opt.ma_win, ...
                        'tolerate_nans', 1);
  col_pale= col_makePale(opt.col, opt.pale_factor);
  H.plot_ma= plot(opt.x, lat_ma, 'Color',col_pale, 'LineWidth',2);
else
  H.plot_ma= [];
end
if opt.plot_density,
  xx= opt.x(1):opt.x(end);
  yy= ismember(xx, opt.x);
  lat_ma= movingAverage(yy', opt.ma_N, ...
                        'method','centered', ...
                        'window', opt.ma_win, ...
                        'tolerate_nans', 1);
  col_pale= col_makePale(opt.col, opt.pale_factor);
  if isfield(ud, 'ax_density'),
    H.ax_density= ud.ax_density;
    axes(H.ax_density);
  else
    H.ax_density= axes('Position',get(H.ax,'Position'));
    set(H.ax_density, 'NextPlot','add');
  end
  H.plot_density= plot(xx, lat_ma, 'Color',col_pale, 'LineWidth',2);
  set(H.ax_density, 'Color','none');
  set(H.ax_density, 'YLim',[0 1], 'YAxisLocation','right', ...
                    'XAxisLocation','top', 'XTick',[]);
  ylabel('density');
  axes(H.ax);
else
  H.plot_density= [];
end
H.plot_lat= plot(opt.x, lat, opt.marker, 'MarkerFaceColor',opt.col, ...
                 opt.opt_plot{:});

if isempty(opt.ylim),
  opt.ylim= ylim;
end
H.blank= [];
if isempty(ud),
  set(H.ax, 'XLim',opt.x([1 end])+[-1 1]*diff(opt.x([1 end]))/50, ...
            'YLim',opt.ylim, 'Color','none');
  if isfield(H, 'ax_density'),
    ud.ax_density= H.ax_density;
    set(H.ax, 'UserData',ud);
    set(H.ax_density', 'XLim', get(H.ax,'XLim'));
  end
  
  if ~isempty(opt.blank),
    for ii= 1:size(opt.blank, 1),
      H.blank(ii)= patch(opt.blank(ii, [1 2 2 1]), opt.ylim([1 1 2 2]), ...
                         opt.blank_color);
    end
    set(H.blank, 'EdgeColor','none');
  end
end

moveObjectForth(H.plot_lat);
xlabel('time  [min]');
ylabel('latency  [ms]');
