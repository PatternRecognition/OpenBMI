function plot_classification_traces(traces, labels, dsply)
%plot_classification_traces(traces, labels, dsply)

if ~isfield(dsply, 'nFigs'), dsply.nFigs=1; end
if ~isfield(dsply, 'color'), 
  dsply.color= [0 1 0.9;  0.28 1 0.7;  0.6 1 0.8;  0.13 1 1;  0.85 1 0.8];
end

traces= squeeze(traces);
nEvents= size(traces, 2);
time_line= dsply.E;

col_h= dsply.color([1:size(labels,1)]*labels, 1);

if ~isfield(dsply, 'yLim'),
  frc= fractileValues(traces(:), 1);
  dsply.yLim= frc([2 4]);
end

inter= floor(linspace(0, nEvents, dsply.nFigs*4+1));
firstFig= gcf;
ii= 0;
for ff= 1:dsply.nFigs,
figure(firstFig+ff-1);
clf;

for ip= 1:4, ii= ii+1;
  evt= inter(ii)+1:inter(ii+1); 
%  evt= ceil(nEvents/4*(ip-1)+1):ceil(nEvents/4*ip);
  ax(ip)= subplotxl(2, 2, ip, [0.05 0.05 0.02], [0.03 0.05 0.02]);

  hold on;
  for cc= 1:size(labels,1),
    ev= find(labels(cc,evt));
    for ie= evt(ev),
      plot(time_line, traces(:,ie), ...
           'color',hsv2rgb([col_h(ie) 0.8 1]), 'lineWidth',0.3);
    end
  end
  hold off;
  set(gca, 'yLim',dsply.yLim);
  ht= title(sprintf('trials %d - %d', evt([1 end])));
  set(ht, 'verticalAli','top');
end  %% for ip
end %% for ff

%% delete yTicks at the border
for ip= 1:4,
  axes(ax(ip));
  yTick= get(gca, 'yTick');
  yTick= setdiff(yTick, get(gca, 'yLim'));
  set(gca, 'yTick',yTick);
end
