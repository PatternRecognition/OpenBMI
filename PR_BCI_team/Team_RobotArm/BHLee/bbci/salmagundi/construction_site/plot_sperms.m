ival= [-350 -100];
ival_mark= -150;
ma= 5;
%% run continuous_classification_loo and continuous_detection3c_loo before
if ma>1,
  warning('applying moving average filter');
end

cd([BCI_DIR 'construction_site'])
load testTraces_lr
lrTraces= outTraces;
lrTraces= movingAverageCausal(lrTraces, ma);
load testTraces_mn3c
mnTraces= outTraces;
mnTraces= movingAverageCausal(mnTraces, ma);

nEvents= size(lrTraces,2);
iv= max(find(E<=ival(1))):min(find(E>=ival_mark));
ivm= max(find(E<=ival_mark)):min(find(E>=ival(2)));

col_h= labels(2,:)/3;

%xLim= 0.018 * [-1 1];
xLim= 1.2 * [-1 1];
yLim= [-1 1.4];
%yLim= [-9 19];  %% for continuous_detection_loo (with QDA)

clf;
for ip= 1:4,
  evt= ceil(nEvents/4*(ip-1)+1):ceil(nEvents/4*ip);
  ax= subplotxl(2, 2, ip, [0.05 0.05 0], [0.03 0.03 0]);
  
  hold on;
  line([-10 10; 0 0]', [0 0; -100 100]', 'color','k');
  for ie= evt,
    plot(lrTraces(iv,ie), mnTraces(iv,ie), ...
         'color',hsv2rgb([col_h(ie) 0.25 1]));
  end
  for ie= evt,
    plot(lrTraces(ivm,ie), mnTraces(ivm,ie), ...
         'color',hsv2rgb([col_h(ie) 0.6 0.8]));
    plot(lrTraces(ivm(end),ie), mnTraces(ivm(end),ie), '.', ...
         'color',hsv2rgb([col_h(ie) 0.8 0.8]));
  end
  set(ax, 'xLim',xLim, 'yLim',yLim);
end
