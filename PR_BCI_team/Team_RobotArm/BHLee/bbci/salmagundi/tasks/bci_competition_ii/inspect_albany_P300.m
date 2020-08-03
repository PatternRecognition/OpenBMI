file= 'bci_competition_ii/albany_P300_train';
fig_dir= 'bci_competition_ii/';

[cnt, mrk, mnt]= loadProcessedEEG(file);
%[epo, mrk, mnt]= loadProcessedEEG(file, 'avg15');

epo= makeEpochs(cnt, mrk, [-50 550]);
epo.code= mrk.code;
%epo= proc_albanyAverageP300Trials(epo, 15);
epo= proc_baseline(epo, [-50 0]);
grid_plot(epo, mnt);
saveFigure([fig_dir 'albany_P300_erp'], [10 7]*2);

clf;
subplot(2,2,[1 2]);
ival= [310 360];
showERP(epo, mnt, {'Fz','FCz','Cz','CPz','Pz'});
xlabel('time [ms]');
ylabel('[\muV]');
grid_markIval(ival);
legend;

topo= proc_meanAcrossTime(epo, ival);
topo= proc_average(topo);
scalp_opt= struct('shading', 'flat');
scalp_opt.resolution= 24;
scalp_opt.contour= 30;
scalp_opt.colAx= [min(topo.x(:)) max(topo.x(:))];
subplot(2,2,3);
plotScalpPattern(mnt, topo.x(:,:,1), scalp_opt);
title(topo.className{1});
subplot(2,2,4);
scalp_opt.contour= 10;
plotScalpPattern(mnt, topo.x(:,:,2), scalp_opt);
title(topo.className{2});
saveFigure([fig_dir 'albany_P300_topo'], [10 7]*2);


dscr_ival= [270 360];
epot= proc_t_scale(epo, 0.01);
tsc_opt.yUnit= '';
tsc_opt.titleAppendix= sprintf('  (\\alpha=%.2f)', epot.alpha);
tsc_opt.colorOrder= [1 0 0];
tsc_opt.scalePolicy= 'sym';
grid_plot(epot, mnt, tsc_opt);
grid_markRange(epot.crit, [], 'color',0.4*[1 1 1]);
grid_markIval(dscr_ival);
saveFigure([fig_dir 'albany_P300_tScaled'], [10 6]*2);


epo_rsqu= proc_r_square(epo);
rsqu_opt.yUnit= '';
rsqu_opt.colorOrder= [1 0 0];
grid_plot(epo_rsqu, mnt, rsqu_opt);
grid_markIval(dscr_ival);
saveFigure([fig_dir 'albany_P300_rsqu'], [10 6]*2);


clf;
topo= proc_meanAcrossTime(epot, dscr_ival);
scalp_opt= struct('shading', 'flat');
scalp_opt.resolution= 24;
scalp_opt.contour= [topo.crit topo.crit];
scalp_opt.colAx= 'range';
subplot(1,2,1);
plotScalpPattern(mnt, topo.x(:,:,1), scalp_opt);
title(topo.className{1});


topo= proc_meanAcrossTime(epo_rsqu, dscr_ival);
scalp_opt.contour= -5;
subplot(1,2,2);
plotScalpPattern(mnt, topo.x(:,:,1), scalp_opt);
title(topo.className{1});
saveFigure([fig_dir 'albany_P300_dscr_topo'], [10 5]*2);


clf;
fv= proc_meanAcrossTime(epo, [310 360], 'Cz');
hist_opt.boundPolicy= 'secondEmptyBin';
plotOverlappingHist(fv, 41, hist_opt);
saveFigure([fig_dir 'albany_P300_hist'], [10 6]*1.3);


fv= proc_commonAverageReference(epo);
fv= proc_baseline(fv, [0 150]);
fv= proc_selectIval(fv, [200 401]);
fv= proc_jumpingMeans(fv, 5);

clf;
colormap(hot(21));
fv= rmfield(fv, 'title');
plot_classification_scalp(fv, mnt, 'LDA');
title(sprintf('<%s> vs. <%s>', fv.className{:}));
saveFigure([fig_dir 'albany_P300_classy_scalp'], [6 5]*2);


opt_series= struct('crossSize',1, 'shading','flat', ...
                   'resolution',15, 'contour',-4);
showERPscalps(epo_av, mnt, -25:25:150, 0, opt_series);
saveFigure([fig_dir 'albany_P300_erp_scalps_series1'], [16 5]*2);
showERPscalps(epo_av, mnt, 175:25:350, 0, opt_series);
saveFigure([fig_dir 'albany_P300_erp_scalps_series2'], [16 5]*2);
showERPscalps(epo_av, mnt, 375:25:550, 0, opt_series);
saveFigure([fig_dir 'albany_P300_erp_scalps_series3'], [16 5]*2);


grd= sprintf('F7,F3,Fz,F4,F8\nT7,C3,Cz,C4,T8\nP7,CP3,CPz,CP4,P8\nPO7,O1,legend,O2,PO8');
mnt_large= setDisplayMontage(mnt, grd);

epo_diff= proc_classDifference(epo, {'deviant','standard'});
grid_plot(epo_diff, mnt_large);
grid_markIval([280 350]);
saveFigure([fig_dir 'albany_P300_diff_erp'], [6 5]*2);

clf;
topo= proc_meanAcrossTime(epo_diff, [280 350]);
plotScalpPattern(mnt, topo.x, scalp_opt);
title(sprintf('%s  [280 350] ms', topo.className{1}));
saveFigure([fig_dir 'albany_P300_diff_topo'], [6 5]*2);

scalp_opt= struct('shading', 'flat');
scalp_opt.resolution= 50;
scalp_opt.contour= [0 0];
showScalpSeries(epo_diff, mnt, 225:25:500, scalp_opt);
saveFigure([fig_dir 'albany_P300_diff_topo_series'], [8 5]*2);
