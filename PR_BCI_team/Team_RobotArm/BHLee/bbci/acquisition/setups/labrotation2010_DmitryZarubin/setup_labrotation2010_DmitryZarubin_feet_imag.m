stim= [];
stim.cue= struct('classes', {'left','right'});
[stim.cue.nEvents]= deal(40);
[stim.cue.timing]= deal([1000 1000 2000]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.handle_background= stimutil_initFigure;
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

desc= stimutil_readDescription('labrotation2010_DmitryZarubin_feet_imag');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'imag_arrow';
opt.breaks= [15 15];  %% Alle 15 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.delete_obj= h_desc.axis;

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');
