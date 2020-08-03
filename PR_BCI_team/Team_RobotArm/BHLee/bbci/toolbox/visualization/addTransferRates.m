function addTransferRates(pace, bTick)
%addTransferRates(pace, bTick)

bTick= -sort(-bTick);

yLim= get(gca, 'yLim');
ha= axes('position', (get(gca,'position')));
set(ha, 'yAxisLocation','right', 'color','none', ...
        'xAxisLocation','top', 'xTick',[]);
p= 1-yLim(2)/100:0.001:1-yLim(1)/100;
rates= 60*1000/pace*bitrate(p);
yTick= zeros(size(bTick));
for ib= 1:length(bTick),
  [mm,mi]= min(abs(rates-bTick(ib)));
  yTick(ib)= 100*(1-p(mi));
end
iok= find(yTick>0);
set(gca, 'yTick',yTick(iok), 'yTickLabel',bTick(iok), 'yLim',rates([1 end]));
ylabel('[bits per minute]');
