N= 15;

opt= struct('perc_dev', 0.5);
opt.response_markers= {'R 16', 'R  8'};
opt.bv_host= 'localhost';
opt.position= [-1919 0 1920 1181];
opt.duration_cue= 2000;
opt.response_delay= [700 1700];
opt.duration_response= 2000;
opt.duration_blank= 1000;
opt.test= 1;

opt.handle_background= stimutil_initFigure(opt);
opt_d2= struct('vpos',0.45, 'd2_fontsize',0.2);
stimutil_showD2cue('d',[0 1 0 1], opt_d2, 'hpos',-0.85);
stimutil_showD2cue('d',[1 1 0 0], opt_d2, 'hpos',-0.30);
stimutil_showD2cue('d',[0 0 1 1], opt_d2, 'hpos', 0.30);
stimutil_showD2cue('d',[1 0 0 1], opt_d2, 'hpos', 0.85);
opt_font= {'Rotation',90, 'FontUnits','normalized', ...
           'FontSize',0.05, 'HorizontalAli','center', 'VerticalAli','bottom'};
text(-1.1, opt_d2.vpos, 'targets: ''J''',opt_font{:});
opt_d2.vpos= -0.55;
stimutil_showD2cue('d',[1 1 1 1], opt_d2, 'hpos',-0.85);
stimutil_showD2cue('d',[0 0 0 0], opt_d2, 'hpos',-0.30);
stimutil_showD2cue('b',[0 1 1 0], opt_d2, 'hpos', 0.30);
stimutil_showD2cue('d',[1 1 0 1], opt_d2, 'hpos', 0.85);
text(-1.1, opt_d2.vpos, 'non-targets: ''F''',opt_font{:});
line([-1 1], [0 0], 'Color','k', 'LineStyle','--');

fprintf('This is just for testing (not recording EEG):\n');
fprintf('stim_d2test(N, opt, ''test'',1);\n');

