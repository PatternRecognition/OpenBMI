function h= plotsigs(z, varargin);

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'color',[]);

[T, nChans]= size(z);
if isempty(opt.color),
  opt.color= cmap_rainbow(nChans);
end

clf;
fac= 1.5*max(abs(z(:)));
for ii= 1:nChans,
  h.p= plot(nChans+1-ii+z(:,ii)/fac, 'Color',opt.color(ii,:));
  hold on;
  h.l= line([-5 T+5], [0 0], 'Color','k');
end
hold off;

set(gca, 'XLim',[-5 T+5], 'YLim',[0.25 5.25], 'YTick',1:4);
