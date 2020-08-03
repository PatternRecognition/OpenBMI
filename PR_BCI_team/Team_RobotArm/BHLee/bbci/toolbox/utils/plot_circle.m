function H= plot_circle(center, rad, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'steps', 360, ...
                  'linespec', {'color','k'});

hold_state= ishold;
hold('on');

T= linspace(0, 2*pi, opt.steps);
xx= rad * cos(T);
yy= rad * sin(T);
H= plot(center(1)+xx, center(2)+yy, opt.linespec{:});

if ~hold_state,
  hold('off');
end
