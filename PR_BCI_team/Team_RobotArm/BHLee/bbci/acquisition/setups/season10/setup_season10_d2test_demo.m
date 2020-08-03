N= 5;

opt= struct('perc_dev', 0.33);
opt.bv_host= 'localhost';
opt.duration_cue= 2000;
opt.response_delay= 1000;
opt.duration_response= 2000;
opt.duration_blank= 1000;
opt.test= 1;

opt.handle_background= stimutil_initFigure;
desc= stimutil_readDescription('vitalbci_season1_d2test');
h_desc= stimutil_showDescription(desc, 'waitfor',0, ...
                                 'desc_pos',[0.5 0.65], ...
                                 'desc_maxsize',[0.9 0.5]);

opt_d2= struct('vpos',-0.8, 'd2_fontsize',0.125);
stimutil_showD2cue('d',[0 1 0 1], opt_d2, 'hpos',-1);
stimutil_showD2cue('d',[1 1 0 0], opt_d2, 'hpos',-.75);
stimutil_showD2cue('d',[0 0 1 1], opt_d2, 'hpos',-.5);
stimutil_showD2cue('d',[1 0 0 1], opt_d2, 'hpos',-.25);
opt_font= {'FontUnits','normalized', ...
           'FontSize',0.05, 'HorizontalAli','center', 'VerticalAli','top'};
text(-0.625, -0.4, 'd2-Ziele: ''J'' drücken',opt_font{:});
stimutil_showD2cue('d',[1 1 1 1], opt_d2, 'hpos',0.25);
stimutil_showD2cue('d',[0 0 0 0], opt_d2, 'hpos',0.5);
stimutil_showD2cue('b',[0 1 1 0], opt_d2, 'hpos',0.75);
stimutil_showD2cue('d',[1 1 0 1], opt_d2, 'hpos',1);
text(0.625, -0.4, 'keine d2-Ziele: ''F'' drücken',opt_font{:});
line([0 0], [-1 -0.4], 'Color','k', 'LineStyle','--', 'LineWidth',2);
