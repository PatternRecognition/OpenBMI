stim= [];
stim.cue= struct('string', {'(L)','(R)','X'});
[stim.cue.nEvents]= deal(15);
[stim.cue.timing]= deal([2000 4000 2000]);

stim.prelude.cue= struct('string', {'L','R'});
[stim.prelude.cue.nEvents]= deal(5);
[stim.prelude.cue.timing]= deal([2000 4000 2000]);

stim.desc= textread([DATA_DIR 'task_descriptions/season7_imag2.txt'],...
                    '%s','delimiter','\n');
stim.msg_intro= 'Vorgestellte Bewegungen';
stim.prelude.msg_intro= 'Ausgeführte Bewegungen';

opt= [];
opt.filename= 'imagwp2_lett';
opt.breaks= [15 15];
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.position= [-1919 -149 1920 1181];

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
