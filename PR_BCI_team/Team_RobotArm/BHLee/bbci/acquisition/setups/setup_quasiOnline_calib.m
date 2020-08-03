stim= [];
stim.cue= struct('string', {'L','R'});
[stim.cue.nEvents]= deal(40);
[stim.cue.timing]= deal([2750 3000; 500 0]);

stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.filename= 'imag_lett';
opt.breaks= [15 15];
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.position= [-1279 0 1280 1005];

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
