function axis_square
%AXIS_SQUARE - Makes the current axis box square in size
%
%Description:
%  The visual effect of this function is the same as that of the matlab
%  command <axis('square')>, but this function changes the axis property
%  'position' instead of the property 'PlotBoxAspectRatio'.
%  This is an advantage in some cases but has the drawback that you cannot
%  undo it.
%
%Usage:
%  axis_square(<AX>)
%
%Input:
%  AX: Handle to the axis, default gca.
%
%See also AXIS.

% Author(s): Benjamin Blankertz, Feb 2005

if nargin==0,
  ax= gca;
end

oldUnits= get(ax, 'units');
set(ax, 'units','normalized');
pos= get(ax, 'position');
oldFigUnits= get(gcf, 'units');
set(gcf, 'units','pixel');
figpos= get(gcf, 'position');
set(gcf, 'units',oldFigUnits);
newpos= pos;
newpos(4)= min(pos([3 4]).*figpos([3 4])/figpos(4));
newpos(2)= pos(2) + (pos(4)-newpos(4))/2;
set(ax, 'position',newpos, 'units',oldUnits);
