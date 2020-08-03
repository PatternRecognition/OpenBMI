stim= [];
stim.cue= struct('classes','speech_eyes_open');
[stim.cue.nEvents]= deal(1);
[stim.cue.timing]= deal([2000 300000 1000; 0 0 0]);
stim.msg_intro= 'Entspannen';
stim.duration_intro= 5000;
stim.msg_extro= 'Entspannen';
stim.duration_extro= 5000;

opt= [];
opt.handle_background= stimutil_initFigure;

desc= stimutil_readDescription('season9_eyes_open');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'eyes_open';
opt.msg_fin= 'Ende';
opt.delete_obj= h_desc.axis;
