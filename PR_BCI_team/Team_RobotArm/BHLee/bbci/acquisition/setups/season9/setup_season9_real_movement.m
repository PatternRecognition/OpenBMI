stim= [];
stim.cue= struct('classes', {'left','right'});
[stim.cue.nEvents]= deal(24);
[stim.cue.timing]= deal([2000 4000 9000; 0 0 3000]);
stim.msg_intro= 'Entspannen';
stim.duration_intro= 10000;
stim.msg_extro= 'Entspannen';
stim.duration_extro= 5000;

opt= [];
opt.handle_background= stimutil_initFigure;
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

desc= stimutil_readDescription('season9_real_movement');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'real_movement';
opt.breaks= [300 15];  %% Alle 12 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.delete_obj= h_desc.axis;

%fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
%fprintf('stim_visualCues(stim, opt);\n');
