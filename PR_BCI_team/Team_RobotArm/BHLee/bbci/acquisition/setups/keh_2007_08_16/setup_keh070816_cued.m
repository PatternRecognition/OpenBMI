stim= [];
stim.cue= struct('string', {'K','Z'});
[stim.cue.nEvents]= deal(50);
[stim.cue.timing]= deal([2000 2000 0]);

stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.filename= 'real_lett';
opt.breaks= [25 15];
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.position= [-1279 0 1280 1005];

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
