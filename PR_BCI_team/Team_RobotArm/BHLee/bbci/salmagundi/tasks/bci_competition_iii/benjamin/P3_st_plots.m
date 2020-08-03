su= 'A';
%su= 'B';

disp_ival_series= [90 150; 195 255; 265 325; 370 430; 440 500];

grid_opt= struct('colorOrder',[1 0 0; 0 0.7 0]);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOGh'}, {'EOGv'}};
grid_opt.scalePolicy= {'auto', 'auto', 'sym', 'auto'};
grid_opt= set_defaults(grid_opt, ...
                       'lineWidth',1, 'axisTitleFontWeight','bold', ...
                       'axisType','cross', 'visible','on', ...
                       'figure_color',[1 1 1]);
scalp_opt= struct('shading','flat', 'resolution',20, ...
                  'colAx','sym', 'colormap',jet(20), ...
                  'contour',-5, 'contour_policy','strict');
scalp_rsq_opt= setfield(scalp_opt, 'colAx','range');
map1= cmap_hsv_fade(10, 1/6, [0 1], 1);
map2= cmap_hsv_fade(11, [1/6 0], 1, 1);
cmap_fire= [map1; map2(2:end,:)];
scalp_rsq_opt.colormap= cmap_fire;

fig_pre= ['preliminary/bci_competition_iii/benjamin/data_set_ii_' su '_'];

[cnt,mrk,mnt]= ...
    loadProcessedEEG([EEG_IMPORT_DIR 'bci_competition_iii/data_set_ii_' su]);

%% prepend 1s
cnt.x= cat(1, repmat(cnt.x(1,:), [cnt.fs 1]), cnt.x);
mrk.pos= mrk.pos + mrk.fs;

idx_dev= find(mrk.y(1,:));
d= [0 diff(idx_dev)];
mrk_dev= mrk_setClasses(mrk, ...
                        {idx_dev(find(d==1)); ...
                         idx_dev(find(d==2)); ...
                         idx_dev(find(d==3)); ...
                         idx_dev(find(d==4)); ...
                         idx_dev(find(d==5)); ...
                         idx_dev(find(d==6))}, ...
                        {'dev1', 'dev2', 'dev3', ...
                         'dev4', 'dev5', 'dev6'});
mrk_dev= mrk_selectEvents(mrk_dev, any(mrk_dev.y));

epo_dev= makeEpochs(cnt, mrk_dev, [-100 650]);
epo_dev= proc_movingAverage(epo_dev, 25, 'centered');
epo_dev= proc_baseline(epo_dev, -epo_dev.t(1), 'trialwise',0);

epo_dev= rmfield(epo_dev, 'refIval');
H= grid_plot(epo_dev, mnt, grid_opt, 'colorOrder','rainbow');
saveFigure([fig_pre 'erp_deviant_dist_noref'], [19 12]);

showERPplusScalpSeries(epo_dev, mnt, 'Pz', disp_ival_series, scalp_opt, ...
                       'colorOrder','rainbow', 'legend_pos','none');
saveFigure([fig_pre 'erp_deviant_dist_noref_plus_topographies'], [15 25]);

epo_dev= proc_baseline(epo_dev, -epo_dev.t(1));
H= grid_plot(epo_dev, mnt, grid_opt, 'colorOrder','rainbow');
saveFigure([fig_pre 'erp_deviant_dist'], [19 12]);

showERPplusScalpSeries(epo_dev, mnt, 'Pz', disp_ival_series, scalp_opt, ...
                       'colorOrder','rainbow', 'legend_pos','none');
saveFigure([fig_pre 'erp_deviant_dist_plus_topographies'], [15 25]);



cnt= proc_selectChannels(cnt, 'F7,z,8','Fp2','FC#','C#','T7,8', ...
                         'CP#','TP7,8','P#','PO#','Oz');
epo_cr= makeEpochs(cnt, mrk, [0 600]);
epo_cr= proc_baseline_cw(epo_cr, [0 100]);

epo_cr.y= [epo_cr.y(1,:) & epo_cr.code<=6; ...
           epo_cr.y(1,:) & epo_cr.code>6; ...
           epo_cr.y(2,:) & epo_cr.code<=6; ...
           epo_cr.y(2,:) & epo_cr.code>6];
epo_cr.className= {'dev col', 'dev row', 'std col', 'std row'};

H= grid_plot(epo_cr, mnt, grid_opt, ...
             'colorOrder', [1 0 0; 1 0.75 0; 0 0.7 0; 0 1 0.75]);
%grid_addBars(epo_rsq, 'h_scale',H.scale);
saveFigure([fig_pre 'erp_colrow'], [19 12]);
