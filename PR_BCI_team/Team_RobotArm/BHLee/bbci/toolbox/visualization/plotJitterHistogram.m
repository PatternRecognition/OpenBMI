function plotJitterHistogram(jit, x, figTitle)
%plotJitterHistogram(jit, x, figTitle)

axes('position', [0.1 0.15 0.8 0.7]);
n= hist(jit, x);
hb= bar(x, n, 'y');
set(gca, 'xLim', [x(1)-10 x(end)+10], 'yLim', [0 1.05*max(n)], ...
         'fontSize',18);
xlabel('time jitter (response time - stimulus time) [ms]');
ylabel('trial count [#]');
title(figTitle);
str{1}= sprintf('mean= %d\\pm%d ms', round(mean(jit)), round(std(jit)));
str{2}= sprintf('median= %g ms', trunc(median(jit)));

text(x(end), max(n), str, 'horizontalAlignment','right', ...
     'verticalAlignment','top', ...
     'fontSize',24);
%     'fontSize',24, 'fontUnit','normalized');
