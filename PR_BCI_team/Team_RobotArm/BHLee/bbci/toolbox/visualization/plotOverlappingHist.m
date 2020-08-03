function plotOverlappingHist(epo_ch, nBins, varargin)
%plotOverlappingHist(epo_ch, nBins, opt)

OPT= propertylist2struct(varargin{:});
OPT= set_defaults(OPT, ...
                  'sum_color', [0.4 0.4 0.4], ...
                  'color', [0 0 1; 0.9 0.6 0], ...
                  'boundPolicy', 'none', ...
                  'xLim', 'auto', ...
                  'xLabel', '', ...
                  'yLabel', '', ...
                  'addToLegend', {});
if ~isfield(OPT, 'title'),
  if isfield(epo_ch, 'title'),
    OPT.title= untex(epo_ch.title); 
  else
    OPT.title= '';
  end
end
if isfield(OPT, 'chan') & ~isempty(OPT.chan),
  epo_ch= proc_selectChannels(epo_ch, OPT.chan);
end

nClasses= size(epo_ch.y,1);
idx= cell(1, nClasses);
if isequal(OPT.xLim, 'auto'),
  for ic= 1:nClasses,
    idx{ic}= find(epo_ch.y(ic,:));
  end
  xx= epo_ch.x([idx{:}]);
  OPT.xLim=[min(xx) max(xx)]; 
else
  xx= epo_ch.x(:)';
  for ic= 1:nClasses,
    idx{ic}= find(epo_ch.y(ic,:) & xx>=OPT.xLim(1) & xx<=OPT.xLim(2));
  end
  xx= xx([idx{:}])';
end

if length(nBins)==1,
  binCenters= linspace(OPT.xLim(1), OPT.xLim(2), nBins);
else
  binCenters= nBins;
end
n= zeros(nClasses, length(binCenters));
for ic= 1:nClasses,
  n(ic,:)= hist(epo_ch.x(idx{ic}), binCenters);
end
nn= sum(n);

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
set(hb, 'faceColor', OPT.sum_color, 'edgeColor', 'none');
hold on;
[so,si]= sort(n);
for jc= nClasses:-1:1,
  for ic= 1:nClasses,
    ii= find(si(ic,:)==jc);
    if ~isempty(ii),
      h= bar(binCenters(ii), n(ic,ii));
      set(h, 'faceColor', OPT.color(ic,:));
      hb(ic)= h;
    end
  end
end
hold off;
axis tight;
xLim= get(gca, 'xLim');
yLim= get(gca, 'yLim');
yLim(2)= yLim(2) + 0.05*diff(yLim);
set(gca, 'yLim', yLim, 'tickDir','out');

xlabel(OPT.xLabel);
ylabel(OPT.yLabel);
title(OPT.title);

legend(hb, cat(2, epo_ch.className, OPT.addToLegend), 2);

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
for ic= 1:nClasses,
  fb= epo_ch.x(idx{ic});
  fb_c1= mean(fb);
  fb_s1= std(fb)/sqrt(length(fb));  %% std of the means
  yy= yLim(1)+diff(yLim)*ic/10;
  str{ic}= sprintf('mean of %s= %.1f \\pm %.1f \\muV', ...
                  epo_ch.className{ic}, fb_c1, std(fb));
  plot(fb_c1+[-1 1]*fb_s1, [yy yy], 'g', 'lineWidth',5);
  plot(fb_c1, yy, 'go','markerSize',12, 'lineWidth',7);
  plot(fb_c1, yy, 'o','markerSize',5, 'lineWidth',5, ...
       'color',OPT.color(ic,:));
end
text(xLim(2)-0.02*diff(xLim), yLim(2)-0.02*diff(yLim), str, ...
     'horizontalAlignment','right', 'verticalAlignment','top', ...
     'fontSize',18, 'fontUnit','normalized');
hold off;
