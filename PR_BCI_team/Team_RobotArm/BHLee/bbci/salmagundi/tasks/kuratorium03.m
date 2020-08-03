file_map= 'Gabriel_01_12_12/selfpaced2sGabriel';
file_erp= 'Gabriel_00_09_05/selfpaced2sGabriel';

fig_dir= 'kuratorium/';


[cnt,mrk,mnt]= loadProcessedEEG(file_map, 'display');

epo= makeSegments(cnt, mrk, [-1500 500]);
epo= proc_baseline(epo, [-1500 -1000]);

erp= proc_selectIval(epo, [-90 0]);
erp= proc_jumpingMeans(erp, 10);
erp= proc_classMean(erp, 1:2);
opt.resolution= 50;
opt.contour= 2;
opt.shading= 'flat';
opt.scalePos= 'none';
[h_ax, h_cb, h_tx]= plotScalpPattern(mnt, erp.x(1,:,2), opt);

dispChans= find(~isnan(mnt.x));
mark= {'C3','C4'};
xx= get(h_tx, 'xData');
yy= get(h_tx, 'yData');
cc= chanind(epo.clab(dispChans), mark);
h= text(xx(cc), yy(cc), mark);
set(h, 'fontSize',14, 'horizontalAlignment','center', 'fontWeight','bold');
hold on;
plot(xx(cc), yy(cc), 'ko', 'markerSize',22, 'lineWidth',2);
hold off;
xx(cc)= NaN;
set(h_tx, 'xData',xx);
saveFigure([fig_dir 'erp_scalp_right'], [8 8], [], 'png');



[cnt,mrk,mnt]= loadProcessedEEG(file_erp, 'display');

chans= {'C3','C4'};
grd= vec2str(chans, '%s', ',');
mnt= setDisplayMontage(mnt, grd);

epo= makeSegments(cnt, mrk, [-1500 500]);
epo= proc_baseline(epo, [-1500 -1300]);

[h_tit, h_ax, h_clab]= showERPgrid(epo, mnt, [-13.5 13.5]);
subtractTitle;
legend('left hand', 'right hand');
set(h_clab,'fontSize', 18);

saveFigure([fig_dir 'erps'], [6 3]*1.5, [], 'png');



left= find(mrk.y(1,:));
right= find(mrk.y(2,:));

ei=ei+1,
ev= [left(ei), right(ei)];
ch= chanind(epo, chans);

for cc= ch,
  axes(h_ax(cc));
  plot(epo.t, squeeze(epo.x(:,cc,ev)));
  set(gca, 'xLim',epo.t([1 end]), 'yTickLabel','');
end
yLim= [-30 30];

[h_tit, h_clab]= showScalpLabels(epo, mnt, 0, yLim, h_ax);
subtractTitle;
%set(h_tit, 'string','single-trials', 'fontSize',0.1);
legend('left hand', 'right hand');
set(h_clab,'fontSize', 18);

saveFigure([fig_dir 'single_trials'], [6 3]*1.5, [], 'png');



file_wheel= 'Gabriel_03_01_15/triggeredsteering2sGabriel';

[cnt, mrk, mnt]= loadProcessedEEG(file_wheel, 'display');
% cnt.x(:,end-1)= 0.1*cnt.x(:,end-1);
grd= 'CCP4';
mnt= setDisplayMontage(mnt, grd);

epo= makeSegments(cnt, mrk, [-600 800]);
epo= proc_baseline(epo, [-600 -400]);

epo_lap= proc_laplace(epo);
%[h_tit, h_ax, h_clab]= showERPgrid(epo_lap, mnt);
clf;
showERP(epo_lap, mnt, 'CCP4');
set(gca, 'xLim',[-500 250], 'yLim',[-2.2 1.2]);
legend('left turn', 'right turn');
shiftAxesDown;

saveFigure([fig_dir 'erp_steering'], [4 3]*1.5, [], 'png');
