function plot_classHistograms(Label, Out, nBins, varargin)

if isstruct(Label),
  Label= Label.y;
end

nClasses= size(Label, 1);

OPT= propertylist2struct(varargin{:});
OPT= set_defaults(OPT, ...
                  'sum_color', [0.4 0.4 0.4], ...
                  'color', [1 0 0; 0 0.7 0], ...
                  'boundPolicy', 'none', ...
                  'xLim', 'auto', ...
                  'xLabel', '', ...
                  'yLabel', '', ...
                  'title', '', ...
                  'legend', cellstr([repmat('class ',nClasses,1) ...
                                     int2str((1:nClasses)')]), ...
                  'addToLegend', {}, ...
                  'showOutputs', 0);

if size(Out,1)>1,
  error('?');
end
nFolds= size(Out,3);
label= repmat(Label, [1 nFolds]);
out= Out(:);

idx= cell(1, nClasses);
if isequal(OPT.xLim, 'auto'),
  for ic= 1:nClasses,
    idx{ic}= find(label(ic,:));
  end
  xx= out([idx{:}]);
  OPT.xLim=[min(xx) max(xx)]; 
else
  xx= out(:)';
  for ic= 1:nClasses,
    idx{ic}= find(label(ic,:) & xx>=OPT.xLim(1) & xx<=OPT.xLim(2));
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
  n(ic,:)= hist(out(idx{ic}), binCenters);
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
  plot_classHistograms(Out, nBins, OPT);
  return;
end

hb= bar(binCenters, nn+0.2, 1);
set(hb, 'faceColor', OPT.sum_color, 'edgeColor', 'none');
hold on;
[so,si]= sort(n);
for jc= nClasses:-1:1,
  for ic= 1:nClasses,
    ii= find(si(jc,:)==ic);
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

legend(hb, cat(2, OPT.legend, OPT.addToLegend), 2);

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
  fb= out(idx{ic});
  fb_c1= mean(fb);
  fb_s1= std(fb)/sqrt(length(fb));  %% std of the means
  yy= yLim(1)+diff(yLim)*ic/10;
  str{ic}= sprintf('mean of class %d= %.1f \\pm %.1f \\muV', ...
                   ic, round(fb_c1), round(std(fb)));
  plot(fb_c1+[-1 1]*fb_s1, [yy yy], 'g', 'lineWidth',5);
  plot(fb_c1, yy, 'yo','markerSize',12, 'lineWidth',7);
  plot(fb_c1, yy, '+','markerSize',9, 'lineWidth',3, 'color',OPT.color(ic,:));
end
text(xLim(2)-0.02*diff(xLim), yLim(2)-0.02*diff(yLim), str, ...
     'horizontalAlignment','right', 'verticalAlignment','top', ...
     'fontSize',18, 'fontUnit','normalized');
hold off;

if OPT.showOutputs,  %% quick and dirty hack
  xLim= get(gca, 'xLim');
  shiftAxesDown(0.5)
  subplot(211)
  cl1= find(Label(1,:));
  cl2= find(Label(2,:));
  for ff= 1:size(Out,3),
    plot(squeeze(Out(1,cl1,ff)), cl1, 'r.'); hold on;
    plot(squeeze(Out(1,cl2,ff)), cl2, '.', 'color',[0 0.7 0]); 
  end
  hold off;
  set(gca, 'xLim',xLim);
end
