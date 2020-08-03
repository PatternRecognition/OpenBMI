su= 'A';
%su= 'B';

switch(su),
 case 'A',
  disp_ival_series= [135 160; 200 250; 300 350; 435 485];
 case 'B',
  disp_ival_series= [175 225; 290 350; 360 400; 450 500];
end


grid_opt= struct('colorOrder',[1 0 0; 0 0.7 0]);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOGh'}, {'EOGv'}};
grid_opt.scalePolicy= {'auto', 'auto', 'sym', 'auto'};
grid_opt= set_defaults(grid_opt, ...
                       'lineWidth',1, 'axisTitleFontWeight','bold', ...
                       'axisType','cross', 'visible','on', ...
                       'figure_color',[1 1 1]);
scalp_opt= struct('shading','flat', 'resolution',50, ...
                  'colAx','sym', 'colormap',jet(20), ...
                  'contour',-5, 'contour_policy','strict');
scalp_rsq_opt= setfield(scalp_opt, 'colAx','range');
map1= cmap_hsv_fade(10, 1/6, [0 1], 1);
map2= cmap_hsv_fade(11, [1/6 0], 1, 1);
cmap_fire= [map1; map2(2:end,:)];
scalp_rsq_opt.colormap= cmap_fire;

fig_pre= ['preliminary/bci_competition_iii/benjamin/data_set_ii_' su '_'];

[epo,mrk,mnt]= ...
    loadProcessedEEG([EEG_IMPORT_DIR 'bci_competition_iii/data_set_ii_' su],...
                     'ave15');

epo= proc_baseline(epo, -epo.t(1));
erp= proc_average(epo);
epo_rsq= proc_r_square(epo);

H= grid_plot(erp, mnt, grid_opt);
grid_addBars(epo_rsq, 'h_scale',H.scale);
saveFigure([fig_pre 'erp'], [19 12]);

showERPplusScalpSeries(erp, mnt, 'Pz', disp_ival_series, scalp_opt, ...
                       'colorOrder',grid_opt.colorOrder, 'legend_pos',2);
grid_addBars(epo_rsq, 'box','on', 'vpos',1);
saveFigure([fig_pre 'erp_plus_topographies'], [20 12]);

epo_rsq.className= {'r^2'};
plotScalpSeries(epo_rsq, mnt, disp_ival_series, scalp_rsq_opt);
saveFigure([fig_pre 'erp_rsq_topographies'], [20 4]);
