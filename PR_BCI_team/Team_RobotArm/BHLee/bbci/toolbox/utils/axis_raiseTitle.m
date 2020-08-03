function ax_raiseTitle(ax, rel)
% AX_RAISETITLE - Raise the title of a plot
%
%Usage:
%  ax_raiseTitle(REL_RAISE)
%  ax_raiseTitle(AX, REL_RAISE)
%
%Input:
%  AX:        Handle of the axis. If not specified, the current axis is used.
%  REL_RAISE: Specifies how much the title is to be raise. The measure is
%             relative to the height of the axis.

%% blanker@first.fhg.de, 01/2005

if nargin==1,
  rel= ax;
  ax= gca;
end

ht= get(ax, 'title');
oldUnits= get(ht, 'units');
set(ht, 'units','normalized');
pos= get(ht, 'position');
set(ht, 'position',pos+[0 rel 0]);
set(ht, 'units',oldUnits);
