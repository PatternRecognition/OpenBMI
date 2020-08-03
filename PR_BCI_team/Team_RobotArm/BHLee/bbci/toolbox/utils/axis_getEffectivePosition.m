function pos= axis_getEffectivePosition(ax)
%AXIS_GETEFFECTIVEPOSITION - Returns position accounting for PlotBoxAspectRatio
%
%Description:
%  The effective position of a plot is not neccessarily reflected by the
%  axis property 'Position'. It can rather be affected by the property
%  'PlotBoxAspectRatio', e.g., after setting axis equal or axis square.
%  This function returns the effective actual position. Please note that
%  the output depends on the acual size of the figure. Also when printing
%  a matlab figure the effective axis position can be different, depending
%  on the figure property 'PaperSize'.
%  Note: At the moment this function only works for 2-D plots.
%
%Status: unstable
%
%Usage:
%  POS= axis_getEffectivePosition(<AX>)
%
%Input:
%  AX: Handle of the axis, default current axis (gca).
%
%Output:
%  POS: position vector [x0 y0 width height], ax the axis property 'Position'.
%
%See also axis.

%Author(s): Benjamin Blankertz, Feb 2005

if nargin==0,
  ax= gca;
end

% Check for 3-D plot.
if all(rem(get(ax,'view'),90)~=0),
  error('does not work for 3-D plots');
end

oldUnits= get(ax, 'units');
set(ax, 'units','pixel');
pospix= get(ax, 'position');
set(ax, 'units','normalized');
posrel= get(ax, 'position');
set(ax, 'units',oldUnits);

pbox= get(ax, 'PlotBoxAspectRatio');
mr= min(pospix([3 4])./pbox([1 2]));

pos([3 4])= pbox([1 2])*mr;
pos([1 2])= pospix([1 2]) + (pospix([3 4])-pos([3 4]))/2;

%% convert to normalized units
pos= pos .* posrel ./ pospix;
