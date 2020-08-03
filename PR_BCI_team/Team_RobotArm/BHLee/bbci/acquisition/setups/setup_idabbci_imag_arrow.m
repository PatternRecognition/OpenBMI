stim= [];
%stim.cue= struct('classes', {'left','right','down'});
stim.cue= struct('classes', {'left','down'});
[stim.cue.nEvents]= deal(40);
[stim.cue.timing]= deal([1500 4000 0]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.position= [1600 0 1280 1005];

opt.handle_background= stimutil_initFigure(opt);
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

opt.filename= 'imag_arrow';
opt.breaks= [20 15];  %% Alle 15 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
