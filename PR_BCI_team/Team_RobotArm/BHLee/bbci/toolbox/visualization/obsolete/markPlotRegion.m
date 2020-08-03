function hp= markPlotRegion(ival, col)
%hp= markPlotRegion(ival, <col>)

bbci_obsolete(mfilename, 'grid_markIval');

if ~exist('col', 'var'), col= 0.9*ones(1, 3); end

xLim= get(gca, 'xLim');
yLim= get(gca, 'yLim');
hp= patch(ival([1 2 2 1]), yLim([1 1 2 2]), col);
set(hp, 'zData',[-1 -1 -1 -1], 'edgeColor','none')
%hl= line(xLim'*ones(1,2), ones(2,1)*yLim);
hl= line(xLim, yLim([1 1]));
hl2= line(xLim, yLim([2 2])-diff(yLim)/1e5);
set([hl hl2], 'color','k');

set(gca, 'yLimMode','manual');
