function plotOverlappingHist(epo_ch, nBins, OPT)
%plotOverlappingHist(epo_ch, nBins, opt)

if ~exist('OPT','var'), OPT.abc= 'def'; end
if ~isfield(OPT, 'color'), OPT.color= [0 0 1; 0.9 0.6 0; 0.4 0.4 0.4]; end
if ~isfield(OPT, 'title'),
  if isfield(epo_ch, 'title'),
    OPT.title= untex(epo_ch.title); 
  else
    OPT.title= '';
  end
end
if ~isfield(OPT, 'boundPolicy'), OPT.boundPolicy='none'; end
if ~isfield(OPT, 'xLim'), OPT.xLim= 'auto'; end
if ~isfield(OPT, 'addToLegend'), OPT.addToLegend={}; end
if isfield(OPT, 'chan') & ~isempty(OPT.chan),
  epo_ch= proc_selectChannels(epo_ch, OPT.chan);
end

if isequal(OPT.xLim, 'auto'),
  ci1= find(epo_ch.y(1,:));
  ci2= find(epo_ch.y(2,:));
  xx= epo_ch.x([ci1 ci2]);
  OPT.xLim=[min(xx) max(xx)]; 
else
  xx= epo_ch.x(:)';
  ci1= find(epo_ch.y(1,:) & xx>=OPT.xLim(1) & xx<=OPT.xLim(2));
  ci2= find(epo_ch.y(2,:) & xx>=OPT.xLim(1) & xx<=OPT.xLim(2));
  xx= xx([ci1 ci2])';
end

if length(nBins)==1,
  binCenters= linspace(OPT.xLim(1), OPT.xLim(2), nBins);
else
  binCenters= nBins;
end
n1= hist(epo_ch.x(ci1), binCenters);
n2= hist(epo_ch.x(ci2), binCenters);
nn= n1+n2;

if strcmp(OPT.boundPolicy, 'secondEmptyBin'),
  z= (nn==0);
  zz= [z & [z(2:end) 0]];
  mz= floor(length(z)/2);
  lb= max([1 find(zz(1:mz))+2]);
  ub= mz + min([find(zz(mz+1:end)) length(z)-mz]-1);
  OPT.xLim= binCenters([lb ub])+[-0.5 0.5]*diff(binCenters([1 2]));
  OPT.addToLegend= {sprintf('%d off limit', sum(nn([1:lb-1 ub+1:end])))};
  OPT.boundPolicy= 'none';
  plotOverlappingHist(epo_ch, nBins, OPT);
  return;
end

hb= bar(binCenters, nn+0.2, 1);
set(hb, 'faceColor', OPT.color(3,:), 'edgeColor', 'none');
hold on;
hb1= bar(binCenters, n1);
set(hb1, 'faceColor', OPT.color(1,:));
hb2= bar(binCenters, n2);
set(hb2, 'faceColor', OPT.color(2,:));
tofront= find(n1<n2);
if ~isempty(tofront),
  hb= bar(binCenters, n1.*(n1<n2));
  set(hb, 'faceColor', OPT.color(1,:));
end
hold off;
axis tight;
xLim= get(gca, 'xLim');
yLim= get(gca, 'yLim');
yLim(2)= yLim(2) + 0.05*diff(yLim);
set(gca, 'yLim', yLim, 'tickDir','out');

if isfield(OPT, 'xLabel'),
  xlabel(OPT.xLabel);
end
if isfield(OPT, 'yLabel'),
  ylabel(OPT.yLabel);
end
title(OPT.title);

legend([hb1, hb2], cat(2, epo_ch.className, OPT.addToLegend), 2);

if isfield(OPT,'fractiles'),
  if OPT.fractiles~=3, error('so far only tertials implemented'); end
  xx= xx(:)';
  [so,si]= sort(xx);
  tertial= xx(si(1+round((length(si)-1)*[1:2]/3)));
  yLim= get(gca, 'yLim');
  hl= line(tertial([1 1; 2 2]'), [yLim; yLim]');
  set(hl, 'color','r', 'lineWidth',2);
  yy= yLim(1)+0.62*diff(yLim);
  text(tertial(1), yy, sprintf('%.1f \\muV \\rightarrow ', tertial(1)), ...
       'horizontalAli','right', 'fontSize',18, 'fontUnit','normalized');
  text(tertial(2), yy, sprintf(' \\leftarrow %.1f \\muV', tertial(2)), ...
       'fontSize',18, 'fontUnit','normalized');
end

hold on;
fb= epo_ch.x(ci1);
fb_c1= mean(fb);
fb_s1= std(fb)/sqrt(length(fb));  %% std of the means
yy= yLim(1)+0.1*diff(yLim);
str{1}= sprintf('mean of %s= %.1f \\pm %.1f \\muV', ...
                epo_ch.className{1}, round(fb_c1), round(std(fb)));
plot(fb_c1+[-1 1]*fb_s1, [yy yy], 'g', 'lineWidth',5);
plot(fb_c1, yy, 'go','markerSize',12, 'lineWidth',7);
plot(fb_c1, yy, 'o','markerSize',5, 'lineWidth',5, 'color',OPT.color(1,:));
fb= epo_ch.x(ci2);
fb_c2= mean(fb);
fb_s2= std(fb)/sqrt(length(fb));
yy= yLim(1)+0.2*diff(yLim);
str{2}= sprintf('mean of %s= %.1f \\pm %.1f \\muV', ...
                epo_ch.className{2}, round(fb_c2), round(std(fb)));
plot(fb_c2+[-1 1]*fb_s2, [yy yy], 'g', 'lineWidth',5);
plot(fb_c2, yy, 'go','markerSize',12, 'lineWidth',7);
plot(fb_c2, yy, 'o','markerSize',5, 'lineWidth',5, 'color',OPT.color(2,:));
text(xLim(2)-0.02*diff(xLim), yLim(2)-0.02*diff(yLim), str, ...
     'horizontalAlignment','right', 'verticalAlignment','top', ...
     'fontSize',18, 'fontUnit','normalized');
hold off;
