stim= [];
stim.cue= struct('bodypart', {'left_hand','right_hand','feet'});
[stim.cue.nEvents]= deal(20);
[stim.cue.timing]= deal([2000 10000 2000]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.handle_background= stimutil_initFigure;

desc= stimutil_readDescription('vitalbci_season1_observation');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'osmr';
opt.breaks= [15 15];  %% Alle 15 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.delete_obj= h_desc.axis;

fprintf('for testing:\n  stim_videoCues(stim, opt, ''test'',1);\n');
fprintf('stim_videoCues(stim, opt);\n');
