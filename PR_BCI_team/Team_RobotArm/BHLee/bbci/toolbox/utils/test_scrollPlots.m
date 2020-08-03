clear all, close all

h1 = figure;
plot(0:0.1:10, sin(0:0.1:10));
title('sine 10')
h2 = figure;

z=peaks(25);
mesh(z);
colormap(hsv)
title('peaks')

h3 = figure;
plot(1:100, sin(0.1:0.1:10), 'r.'); hold on
plot(1:100, sin(1:100));
title('sines')

showColorbar = [0 1 0];
sp_addPlots({h1, h2, h3}, showColorbar);  % variable 'showColorbar' is optional
sp_addLegends({}, {}, {'sin(0.1:0.1:10)', 'sin(1:100)'})
scrollPlots();

% adding plots after scrollPlots was called is also possible!
h4 = figure;
imagesc(randn(4,5))
title('random image')
sp_addPlots(h4, 1)


