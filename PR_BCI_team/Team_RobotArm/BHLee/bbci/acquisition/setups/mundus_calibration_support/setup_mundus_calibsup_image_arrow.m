function [stim, opt]=setup_mundus_calibsup_image_arrow(classes, nEvents, timing_pause, timing_arrow)

stim= [];

stim.cue= struct('classes', {'left','right','down'});

[stim.cue.nEvents]= deal(nEvents);
[stim.cue.timing]= deal([timing_pause timing_arrow timing_pause]);
stim.msg_intro= 'Gleich geht''s los';

opt= [];
opt.classesToUse = classes;
%--------------------------------------------------------------------------
% uncomment to test the arrows!!!!
%--------------------------------------------------------------------------
%opt.test_mode = 1;
opt.handle_background= stimutil_initFigure;
[H, opt.handle_cross]= stimutil_cueArrows({stim.cue.classes}, opt, 'cross',1);
Hc= num2cell(H);
[stim.cue.handle]= deal(Hc{:});

desc= stimutil_readDescription('vitalbci_season1_imag_arrow');
h_desc= stimutil_showDescription(desc, 'waitfor',0);

opt.filename= 'imag_arrow';
opt.breaks= [15 15];  %% Alle 15 Stimuli Pause für 15 Sekunden
opt.break_msg= 'Kurze Pause (%d s)';
opt.msg_fin= 'Ende';
opt.delete_obj= h_desc.axis;

fprintf('for testing:\n  stim_visualCues(stim, opt, ''test'',1);\n');
fprintf('stim_visualCues(stim, opt);\n');

end