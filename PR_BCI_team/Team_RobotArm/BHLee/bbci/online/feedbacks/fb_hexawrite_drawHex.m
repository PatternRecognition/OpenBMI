function h= fb_hexawrite_drawHex(center, angle0, varargin)

opt= propertylist2struct(varargin{:});

hold on;
outline= zeros(2, 6);
hexangles= angle0:2*pi/6:angle0+2*pi;
if length(opt.label)<6,
  opt.label(6)= ' ';
end
for ai= 1:6,
  angie= hexangles(ai);
  dx= opt.hexradius*sin(angie);
  dy= opt.hexradius*cos(angie);
  outline(:,ai)= center + [dx; dy];
  tx= opt.hexradius*opt.label_radfactor*sin(angie+2*pi/12);
  ty= opt.hexradius*opt.label_radfactor*cos(angie+2*pi/12);
  h.label(ai)= text(tx+center(1), ty+center(2), opt.label(ai));
end
set(h.label, 'HorizontalAli','center', 'VerticalAli','middle', ...
	      'FontUnits', 'normalized', opt.label_spec{:});

angie= angle0 - 2*pi/12;
hex_h= opt.hexradius/2/tan(2*pi/12);
arccenter= center + 2*hex_h*[sin(angie); cos(angie)];
arcangle0= angle0 + pi;
arcangles= linspace(arcangle0, arcangle0 - 2*pi/6, opt.arcsteps);
for ai= 1:length(arcangles),
  angie= arcangles(ai);
  dx= opt.hexradius*sin(angie);
  dy= opt.hexradius*cos(angie);
  outline(:,6+ai)= arccenter + [dx; dy];
end
h.outline= plot(outline(1,:), outline(2,:), 'Color','k', opt.hex_spec{:});
hold off;
