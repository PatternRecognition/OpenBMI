stim= [];
[stim.cue.nEvents]= deal(50);
[stim.cue.timing]= deal([2000 2000 0]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.position= [-1279 0 1280 1005];

opt.handle_background= stimutil_initFigure(opt);

H= stimutil_drawPicture({'RechteHandRuhe.bmp', ...
                         'RechteHandZeigefinger.bmp',...
                         'RechteHandKleinerFinger.bmp'}, ...
                        'pic_dir', 'stimuli/');
stim.cue.handle= H.image([2 3]);
opt.handle_cross= H.image(1);

opt.filename= 'real_handcues';
opt.breaks= [25 15];
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
