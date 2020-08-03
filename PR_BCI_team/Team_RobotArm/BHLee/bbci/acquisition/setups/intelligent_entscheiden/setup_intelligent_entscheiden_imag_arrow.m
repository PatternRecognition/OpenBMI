stim= [];
stim.cue= struct('classes', {'left','right'});
[stim.cue.nEvents]= deal(25);
[stim.cue.timing]= deal([2000 4000 2000]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.handle_background= stimutil_initFigure;
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

desc= stimutil_readDescription('vitalbci_season1_imag_arrow');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'imag_arrow';
opt.breaks= [20 15];  %% Alle 15 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.delete_obj= h_desc.axis;

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
