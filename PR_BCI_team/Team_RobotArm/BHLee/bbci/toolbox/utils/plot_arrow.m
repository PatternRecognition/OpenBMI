function hp= plot_arrow(xx, varargin)
%PLOT_ARROW - plot an arrow
%
%Description:
%  This function plots an arrow, (or several arrows at the same time).
%
%Usage:
%  H= PLOT_ARROW(XY, <OPT>)
%  H= PLOT_ARROW(XX, YY, <OPT>)
%
%Input:
%  XX: X-positions, start values in the first column, end values in the second.
%     If XX is just a row vector, the origin (0,0) is set as start point.
%     In this case also YY is assumed to be a row vector.
%  YY: Y-positions, start values in the first column, end values in the second.
%  XY: X-positions in the first, Y-positions in the second row. Vectors
%     start from the oprigin (0,0).
%  OPT: property/value list or struct of optional properties:
%    .lineWidth (1), 
%    .lineStyle ('-'), 
%    .color ([0 0 0]): determine how the line part is drawn
%    .edgeColor ([0 0 0]),
%    .faceColor ([1 1 1]): determine how the head part is drawn
%    .arrowWidth (0.08),
%    .arrowLength (0.12),
%    .arrowUnit ('relative'): determine the size of the head part
%    .arrowType (2): determines the look of the head (to be extended!)
%
%Output:
%  H:  Struct of handle to the graphic objects
%
%Note:
%  - The head of the arrow looks only decent for axis equal.
%  - As .arrowUnit so far only 'relative' is implemented (which
%  should more consistently be called 'normalized').
%
%Example:
%  plot_arrow([0.1 0.3; 0.4 0.7]', [0.1 0.5; 0.2 0.8]', ...
%    'lineWidth',3, 'faceColor',[1 0 0]); axis equal;


if ~isempty(varargin) & isnumeric(varargin{1}) & ~isempty(varargin{1}),
  yy= varargin{1};
  varargin= varargin(2:end);
else
  if size(xx,1)~=2,
    error('input format does not match my expectancy');
  end
  yy= xx(2,:);
  xx= xx(1,:);
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'lineWidth', 1, ...
                  'lineStyle', '-', ...
                  'color', [0 0 0], ...
                  'edgeColor', [0 0 0], ...
                  'faceColor', [1 1 1], ...
                  'arrowWidth', 0.08, ...
                  'arrowLength', 0.12, ...
                  'arrowUnit', 'relative', ...
                  'arrowType', 2);

if ~isequal(opt.arrowUnit, 'relative'),
  error('only arrowUnit ''relative'' is implemented so far');
end

if size(xx,1)==1,
  xx= [zeros(size(xx)); xx];
  yy= [zeros(size(yy)); yy];
end

if prod(size(xx))>length(xx),
  for ii= 1:size(xx,2),
    hp(ii)= plot_arrow(xx(:,ii), yy(:,ii), opt);
  end
  if nargout==0, clear hp; end
  return;
end

hp.line= line(xx, yy);
set(hp.line, 'lineWidth',opt.lineWidth, 'color',opt.color, ...
             'lineStyle',opt.lineStyle);

v= [diff(xx) diff(yy)];
vo= [-v(2) v(1)];
zz= [xx(2) yy(2)] - v*opt.arrowLength;
p1= zz + vo*opt.arrowWidth/2;
p2= zz - vo*opt.arrowWidth/2;

switch(opt.arrowType),
 case 1,
  hp.arrowHead= line([p1(1);xx(2);p2(1)], [p1(2);yy(2);p2(2)]);
  set(hp.arrowHead, 'color',opt.color, 'lineWidth',opt.lineWidth);
 case 2,
  hp.arrowHead= patch([xx(2) p1(1) p2(1)], [yy(2) p1(2) p2(2)], opt.faceColor);
  set(hp.arrowHead, 'edgeColor',opt.edgeColor, 'lineWidth',opt.lineWidth);
 otherwise
  error('unknown arrow type');
end

if nargout==0,
  clear hp;
end
