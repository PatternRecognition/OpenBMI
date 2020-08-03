function H= plot_hist_with_subdistrib(x, idx, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'color', [.5 .5 .5], ...
                  'range', 'auto', ...
                  'xlim', 'auto', ...
                  'ylim', 'auto', ...
                  'percent', 1, ...
                  'gap', 1/8, ...
                  'bins', max(10, min(50, length(x)/10)), ...
                  'legend', []);

nSubs= length(idx);
if size(opt.color,1)==1,
  opt.color= cat(1, color, cmap_rainbow(nSubs));
end

if isequal(opt.range, 'auto'),
  opt.range= [min(x) max(x)];
end

if length(opt.bins)==1,
  if diff(opt.range)==0,
    bincenters= opt.range(1);
  else
    bincenters= linspace(opt.range(1), opt.range(2), opt.bins);
  end
else
  bincenters= opt.bins;
end

if length(bincenters)==1,
  bd= 1;
  n= [length(idx{1}); length(idx{2})];    
else
  bd= mean(diff(bincenters));
  n= zeros(nSubs, length(bincenters));
  for is= 1:nSubs,
    n(is,:)= hist(x(idx{is}), bincenters);
  end
end
nn= sum(n, 1);
if opt.percent,
  n= 100*n/sum(nn);
  nn= 100*nn/sum(nn);
end

cla;
H.histsum= bar(bincenters, nn, 1);
set(H.histsum, 'FaceColor',opt.color(1,:));
hold on;
barwidth= (1-opt.gap)/nSubs;
for is= 1:nSubs,
%  H.hist(is)= bar(bincenters, max(0, n(is,:)-0.001*max(nn)), barwidth);
  H.hist(is)= bar(bincenters, n(is,:), barwidth);
  set(H.hist(is), 'FaceColor',opt.color(1+is,:), 'EdgeColor','none');
  xData= get(H.hist(is), 'XData');
  f= sign((nSubs+1)/2-is);
  xData= xData + bd * ([is-1.5]/nSubs + f*opt.gap/4);
  set(H.hist(is), 'XData',xData);
end
H.histsumframe= bar(bincenters, nn, 1);
set(H.histsumframe, 'FaceColor','none');
hold off;

if isequal(opt.xlim, 'auto'),
  i1= find(nn>0, 1, 'first');
  i2= find(nn>0, 1, 'last');
  opt.xlim= bincenters([i1 i2]) + [-1 1]*bd*(0.5+0.1);
end
if isequal(opt.ylim, 'auto'),
  opt.ylim= [0 max(nn)*1.1];
end
set(gca, 'XLim',opt.xlim, 'YLim',opt.ylim, 'TickDir','out');
if opt.percent,
  ylabel('[%]');
end
if ~isempty(opt.legend),
  H.legend= legend(H.hist, opt.legend);
end
