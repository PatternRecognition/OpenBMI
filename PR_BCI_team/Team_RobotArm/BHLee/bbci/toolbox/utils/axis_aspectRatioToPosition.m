function axis_aspectRatioToPosition(ax)

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

pbox= get(ax, 'PlotBoxAspectRatio');
mr= min(pos([3 4])./pbox([1 2]));

newpos([3 4])= pbox([1 2])*mr;
newpos([1 2])= pos([1 2]) + (pos([3 4])-newpos([3 4]))/2;

set(ax, 'position',newpos);
set(ax, 'units',oldUnits);
