function axisequalwidth(ax)

if ~exist('ax','var'), ax=gca; end

%lims= axis;
axis tight;
xRange= get(gca, 'xLim');
yRange= get(gca, 'yLim');
%axis(lims);

width= max(diff(xRange), diff(yRange));
ww= [-1 1]*width/2;
set(gca, 'xLim',mean(xRange)+ww);
set(gca, 'yLim',mean(yRange)+ww);
