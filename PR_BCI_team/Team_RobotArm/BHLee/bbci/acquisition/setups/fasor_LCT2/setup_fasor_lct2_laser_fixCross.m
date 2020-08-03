stim= [];
stim.cue= struct('classes', {'left'});
[stim.cue.marker]= deal(64);
[stim.cue.nEvents]= deal(60); % ungefähr 10 Minuten
[stim.cue.timing]= deal([10000 500 0]);  % [InterTrialInterval LaserDuration CrossOff]
[stim.cue.jitter]= deal(5000);
stim.msg_intro= 'REINE REAKTIONSAUFGABE (ca. 10 Min)';

opt= [];
opt.duration_intro = 3000;
opt.handle_background= stimutil_initFigure(opt);
opt.cross_vpos =-0.05;
H= stimutil_cueArrows({stim.cue.classes}, opt);
Hc= num2cell(H);
stim.cue.handle=deal(Hc{:});

desc= stimutil_readDescription('fasor_LCT2_K0');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'fasor_LCT2_K0_';
opt.msg_fin= 'Ende dieses Blocks';
opt.delete_obj= h_desc.axis;
opt.bv_host= ''; % 

fprintf('Testen mit :\n  stim_laserCues_BackgroundSound(stim, opt, ''test'',1);\n');
fprintf('Sonst :\n  stim_laserCues_BackgroundSound(stim, opt);\n');
