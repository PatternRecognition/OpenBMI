function H= stimutil_drawArrow(direction, varargin)
%STIMUTIL_DRAWARROW - Draw a thick arrow
%
%Synopsis:
% H= stimutil_drawArrow(DIRECTION, 'Prop1', VALUE1, ...)
%
%Arguments:
% DIRECTION: Derection of the arrow. Format is either a scalar
%   corresponding to a clock (3, 6, 9, 12), or a string ('right',
%   'down', 'left', 'up').
% Optional Properties:
%  'arrow_size': Size of the arrow in the units of the axis
%  'arrow_width': Width of the arrow's line, relative to arrow_size.
%  'arrow_blunt': Width of the arrow's blunt edges, relative to arrow_size.
%     Use 0 for sharp edges. Default: 0 if 'cross'=0, else 0.1.
%  'arrow_color': Default: 0.8*[1 1 1]
%  'arrow_edgecolor': Default 'none'.
%  'cross': Switch to superimpose a fiaxtion cross; Default: 0
%  'cross_size': Size of the cross, relative to arrow_size
%  'cross_width': Width of the cross' line, relative to arrow_size
%  'cross_color': Default: Color of the figure.
%  'cross_edgecolor': Default: 'none'.
%
%Returns
% H: Struct of handles of the graphical objects

% blanker@cs.tu-berlin.de, Nov-2007

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'arrow_size', 0.3, ...
                 'arrow_width', 0.2, ...
                 'arrow_blunt', 0, ...
                 'arrow_tip_blunt', 0, ...
                 'arrow_color', 0.8*[1 1 1], ...
                 'arrow_edgecolor', 'none', ...
                 'cross', 0, ...
                 'cross_size', 0.2, ...
                 'cross_width', 0.03, ...
                 'cross_color', 0*[1 1 1], ...
                 'cross_edgecolor', 'none');

x= opt.arrow_size;
w= x*opt.arrow_width;
b= x*opt.arrow_blunt;
t= x*opt.arrow_tip_blunt;

%xpr= [0 x 0 0 -x+w -x+w 0];
%ypr= [x 0 -x -w -w w w];
xpr= [b x x b -b -b -x+w -x+w -b -b];
ypr= [x t -t -x -x -w -w w w x];
switch(direction),
 case {'up','tongue',0,12},
  xp= ypr;
  yp= xpr;
 case {'right',3},
  xp= xpr;
  yp= ypr;
 case {'down','foot',6},
  xp= -ypr;
  yp= -xpr;
 case {'left',9},
  xp= -xpr;
  yp= -ypr;
 otherwise,
  error('invalid direction: use 3, 6, 9, 12; or ''right'', ''down'', ...');
end

H.arrow= patch(xp, yp, opt.arrow_color);
set(H.arrow, 'EdgeColor', opt.arrow_edgecolor);

if opt.cross,
  c= x*opt.cross_size;
  v= x*opt.cross_width;
  H.cross= patch([v v c c v v -v -v -c -c -v -v], ...
                 [c v v -v -v -c -c -v -v v v c], opt.cross_color);
  set(H.cross, 'EdgeColor', opt.cross_edgecolor);
end
