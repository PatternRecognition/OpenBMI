N= 200;
cues= {'left','right'}; 

opt= struct('perc_dev', 0.15);
opt.response_markers= {'R  8', 'R 16'};
opt.bv_host= 'localhost';
opt.isi= 1600;
opt.duration_cue= 200;
opt.position= [-1919 0 1920 1181];
opt.cross= 1;

opt.handle_background= stimutil_initFigure(opt);
[H, opt.handle_cross]= stimutil_cueArrows(cues, opt);
opt.cue_dev= H(1);
opt.cue_std= H(2);

fprintf('for testing:\n  stim_oddballVisual(N, opt, ''test'',1);\n');
fprintf('stim_oddballVisual(N, opt);\n');
