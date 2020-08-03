toc= -120;             %% time-point of classification
min_ival= [-1580 0] + toc;

ma= 1;
%% run continuous_classification_loo and continuous_detection3c_loo before
if ma>1,
  warning('applying moving average filter');
end

load([BCI_DIR 'construction_site/cronTraces_lr']);
lrTraces= outTraces;
lrTraces= movingAverageCausal(lrTraces, ma);
load([BCI_DIR 'construction_site/cronTraces_mn3c']);
mnTraces= outTraces;
mnTraces= movingAverageCausal(mnTraces, ma);

nEvents= size(lrTraces,2);
iv_min= max(find(E<=min_ival(1))):min(find(E>=min_ival(2)));
idx_toc= max(find(E<=toc));

col_h= labels(2,:)/3;

%xLim= 0.018 * [-1 1];
xLim= 1.49 * [-1 1];
yLim= 1.49 * [-1 1];

clf;
for ip= 1:4,
  evt= ceil(nEvents/4*(ip-1)+1):ceil(nEvents/4*ip);
  ax= subplotxl(2, 2, ip, [0.05 0.05 0], [0.03 0.03 0]);
  
  hold on;
  line([-10 10; 0 0]', [0 0; -100 100]', 'color','k');
  set(ax, 'xLim',xLim, 'yLim',yLim);
  for ie= evt,
    [mi, ix_min]= min(mnTraces(iv_min,ie));
    idx_min= iv_min(ix_min);
    min_time(ie)= E(idx_min);
    plot_vector(lrTraces([idx_min idx_toc],ie), ...
                mnTraces([idx_min idx_toc],ie), ...
                'color',hsv2rgb([col_h(ie) 0.8 0.8]));
    plot(lrTraces(idx_toc,ie), mnTraces(idx_toc,ie), '.', ...
         'color',hsv2rgb([col_h(ie) 0.8 0.8]));
  end
end

saveFigure('divers/fb_vectors');

clf;
hist(min_time, 20);
%hist(min_time, E(iv_min(1):idx_toc));
saveFigure('divers/fb_vectors_mini_hist');
