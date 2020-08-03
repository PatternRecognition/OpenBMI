function axis_equal(ax)
%AXIS_EQUAL - Makes the current axis box square in size
%
%Description:
%  The visual effect of this function is the same as that of the matlab
%  command <axis('equal')>, but this function changes the axis property
%  'position' instead of the properties 'DataAspectRatio' and
%  'PlotBoxAspectRatio'.
%  This is an advantage in some cases but has the drawback that you cannot
%  undo it.
%
%Usage:
%  axis_equal(<AX>)
%
%Input:
%  AX: Handle to the axis, default gca.
%
%See also AXIS.

% Author(s): Benjamin Blankertz, Feb 2005

if nargin==0,
  ax= gca;
end

% Check for 3-D plot.
if all(rem(get(ax,'view'),90)~=0),
  error('does not work for 3-D plots');
end

oldUnits= get(ax, 'units');
set(ax, 'units','pixel');
pos= get(ax, 'position');
d(1)= diff(get(ax, 'xLim'));
d(2)= diff(get(ax, 'yLim'));
mr= min(pos([3 4])./d);

newpos([3 4])= d*mr;
newpos([1 2])= pos([1 2]) + (pos([3 4])-newpos([3 4]))/2;
set(ax, 'position',newpos);
set(ax, 'units',oldUnits);
